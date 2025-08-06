#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Database client implementation with dual strategy.
- Supabase as primary database for production
- SQLite as fallback for development/cache
- Connection validation and error handling
"""

import os
import sqlite3
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
import json
import threading

# .env 자동 로드
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from supabase import create_client, Client
except ImportError:
    Client = None

try:
    from config import config
except ImportError:
    # 기본 설정값 제공
    class DefaultConfig:
        def get_database_config(self):
            return {
                'supabase_url': '',
                'supabase_key': '',
                'sqlite_path': 'newsbot.db'
            }
    config = DefaultConfig()

logger = logging.getLogger(__name__)

class DatabaseClient:
    """Dual database strategy implementation."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'DatabaseClient':
        """Singleton pattern for database client."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self.supabase_client: Optional[Client] = None
        self.sqlite_path = config.get_database_config()['sqlite_path']
        
        # REQUIRED: Initialize connections
        self._initialize_supabase()
        self._initialize_sqlite()
        self._validate_connections()
    
    def _initialize_supabase(self) -> None:
        """Initialize Supabase client if available."""
        try:
            db_config = config.get_database_config()
            supabase_url = db_config['supabase_url']
            supabase_key = db_config['supabase_key']
            
            if supabase_url and supabase_key and Client:
                self.supabase_client = create_client(supabase_url, supabase_key)
                logger.info("Supabase client initialized successfully")
            else:
                logger.warning("Supabase credentials not available, using SQLite only")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
            self.supabase_client = None
    
    def _initialize_sqlite(self) -> None:
        """Initialize SQLite database with required tables."""
        try:
            os.makedirs(os.path.dirname(self.sqlite_path), exist_ok=True)
            
            with sqlite3.connect(self.sqlite_path) as conn:
                cursor = conn.cursor()
                
                # REQUIRED: Create user_profiles table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_profiles (
                        user_id TEXT PRIMARY KEY,
                        profile_json TEXT NOT NULL,
                        summary TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                ''')
                
                # REQUIRED: Create chat_history table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS chat_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        agent TEXT
                    )
                ''')
                
                # REQUIRED: Create portfolio_recommendations table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS portfolio_recommendations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        recommendation_json TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                ''')
                
                conn.commit()
                logger.info("SQLite database initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize SQLite database: {e}")
            raise
    
    def _validate_connections(self) -> None:
        """Validate database connections on startup."""
        # REQUIRED: Test Supabase connection
        if self.supabase_client:
            try:
                self.supabase_client.table('user_profiles').select('user_id').limit(1).execute()
                logger.info("Supabase connection validated")
            except Exception as e:
                logger.warning(f"Supabase connection validation failed: {e}")
                self.supabase_client = None
        
        # REQUIRED: Test SQLite connection
        try:
            with sqlite3.connect(self.sqlite_path) as conn:
                conn.execute("SELECT 1")
            logger.info("SQLite connection validated")
        except Exception as e:
            logger.error(f"SQLite connection validation failed: {e}")
            raise

class UserProfileService:
    """Service layer for user profile operations."""
    
    def __init__(self):
        self.db_client = DatabaseClient()
        self.cache: Dict[str, Dict[str, Any]] = {}
    
    def save_user_profile(self, user_id: str, profile_data: Dict[str, Any]) -> bool:
        """
        Save user profile with validation and encryption.
        
        Args:
            user_id: Unique user identifier
            profile_data: User profile data dictionary
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # REQUIRED: Data validation
            if not user_id or not isinstance(profile_data, dict):
                raise ValueError("Invalid user_id or profile_data")
            
            # REQUIRED: Sanitize data
            sanitized_data = self._sanitize_profile_data(profile_data)
            
            profile_json = json.dumps(sanitized_data, ensure_ascii=False)
            timestamp = datetime.now().isoformat()
            summary = sanitized_data.get('overall_analysis', '')
            
            # REQUIRED: Try Supabase first
            if self.db_client.supabase_client:
                try:
                    result = self.db_client.supabase_client.table("user_profiles").upsert({
                        "user_id": user_id,
                        "profile_json": profile_json,
                        "summary": summary,
                        "created_at": timestamp,
                        "updated_at": timestamp
                    }).execute()
                    
                    logger.info(f"User profile saved to Supabase: {user_id}")
                except Exception as e:
                    logger.warning(f"Supabase save failed, falling back to SQLite: {e}")
            
            # REQUIRED: SQLite fallback/cache
            with sqlite3.connect(self.db_client.sqlite_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO user_profiles 
                    (user_id, profile_json, summary, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                ''', (user_id, profile_json, summary, timestamp, timestamp))
                conn.commit()
            
            # REQUIRED: Cache update
            self.cache[user_id] = sanitized_data
            
            logger.info(f"User profile saved successfully: {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save user profile {user_id}: {e}")
            return False
    
    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user profile with caching and fallback.
        
        Args:
            user_id: Unique user identifier
            
        Returns:
            User profile data or None if not found
        """
        try:
            # REQUIRED: Check cache first
            if user_id in self.cache:
                return self.cache[user_id]
            
            profile_json = None
            
            # REQUIRED: Try Supabase first
            if self.db_client.supabase_client:
                try:
                    result = self.db_client.supabase_client.table("user_profiles").select(
                        "profile_json,summary"
                    ).eq("user_id", user_id).order(
                        "created_at", desc=True
                    ).limit(1).execute()
                    
                    if result.data and len(result.data) > 0:
                        profile_json = result.data[0]["profile_json"]
                        logger.info(f"User profile retrieved from Supabase: {user_id}")
                except Exception as e:
                    logger.warning(f"Supabase retrieval failed, trying SQLite: {e}")
            
            # REQUIRED: SQLite fallback
            if not profile_json:
                with sqlite3.connect(self.db_client.sqlite_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT profile_json FROM user_profiles 
                        WHERE user_id = ? ORDER BY updated_at DESC LIMIT 1
                    ''', (user_id,))
                    
                    row = cursor.fetchone()
                    if row:
                        profile_json = row[0]
                        logger.info(f"User profile retrieved from SQLite: {user_id}")
            
            if profile_json:
                # profile_json이 이미 dict인 경우 처리
                if isinstance(profile_json, dict):
                    profile_data = profile_json
                else:
                    profile_data = json.loads(profile_json)
                
                # REQUIRED: Update cache
                self.cache[user_id] = profile_data
                
                return profile_data
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get user profile {user_id}: {e}")
            return None
    
    def _sanitize_profile_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize profile data to prevent XSS and injection.
        
        Args:
            data: Profile data to sanitize
            
        Returns:
            Sanitized profile data
        """
        try:
            import bleach
        except ImportError:
            logger.warning("bleach not available, skipping HTML sanitization")
            return data
        
        sanitized = {}
        for key, value in data.items():
            if isinstance(value, str):
                # REQUIRED: Strip HTML tags and sanitize
                sanitized[key] = bleach.clean(value, tags=[], strip=True)
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_profile_data(value)
            else:
                sanitized[key] = value
        
        return sanitized

class ChatHistoryService:
    """Service layer for chat history operations."""
    
    def __init__(self):
        self.db_client = DatabaseClient()
    
    def save_chat_message(self, session_id: str, role: str, content: str, agent: Optional[str] = None) -> bool:
        """
        Save chat message to database.
        
        Args:
            session_id: User session identifier
            role: Message role (user/assistant)
            content: Message content
            agent: AI agent name if applicable
            
        Returns:
            True if saved successfully
        """
        try:
            timestamp = datetime.now().isoformat()
            
            # REQUIRED: SQLite storage
            with sqlite3.connect(self.db_client.sqlite_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO chat_history (session_id, role, content, timestamp, agent)
                    VALUES (?, ?, ?, ?, ?)
                ''', (session_id, role, content, timestamp, agent))
                conn.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save chat message: {e}")
            return False
    
    def get_chat_history(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get chat history for session.
        
        Args:
            session_id: User session identifier
            limit: Maximum number of messages to retrieve
            
        Returns:
            List of chat messages
        """
        try:
            with sqlite3.connect(self.db_client.sqlite_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT role, content, timestamp, agent FROM chat_history 
                    WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?
                ''', (session_id, limit))
                
                rows = cursor.fetchall()
                return [
                    {
                        "role": row[0],
                        "content": row[1],
                        "timestamp": row[2],
                        "agent": row[3]
                    }
                    for row in reversed(rows)  # Return in chronological order
                ]
                
        except Exception as e:
            logger.error(f"Failed to get chat history: {e}")
            return []

def get_supabase_client() -> Optional[Client]:
    """Get Supabase client instance."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        # Supabase가 설정되지 않은 경우 None 반환
        logger.warning("Supabase 환경변수가 설정되지 않았습니다. SQLite 폴백 모드로 작동합니다.")
        return None
    try:
        return create_client(url, key)
    except Exception as e:
        logger.error(f"Supabase 클라이언트 생성 실패: {e}")
        return None

def get_user_profile_service() -> UserProfileService:
    """Get user profile service instance."""
    return UserProfileService()

def get_chat_history_service() -> ChatHistoryService:
    """Get chat history service instance."""
    return ChatHistoryService() 