#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Memory management for multi-agent AI conversations.
- Separate memory streams for user-AI and AI-AI conversations
- Context caching and session management
- Conversation history management
- Memory cleanup and optimization
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import threading

logger = logging.getLogger(__name__)

class MemoryManager:
    """Manages conversation memory for multi-agent AI system."""
    
    def __init__(self):
        # REQUIRED: User-AI conversation (shown to user)
        self.user_ai_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # REQUIRED: AI-AI conversation (internal coordination)
        self.ai_ai_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # REQUIRED: Context cache
        self.context_cache: Dict[str, Dict[str, Any]] = {}
        
        # REQUIRED: Session metadata
        self.session_metadata: Dict[str, Dict[str, Any]] = {}
        
        # REQUIRED: Thread safety
        self._lock = threading.Lock()
        
        logger.info("MemoryManager initialized successfully")
    
    def add_user_ai_message(self, session_id: str, role: str, content: str, agent: Optional[str] = None) -> None:
        """
        Add message to user-AI conversation stream.
        
        Args:
            session_id: User session identifier
            role: Message role ('user' or 'assistant')
            content: Message content
            agent: AI agent name if applicable
        """
        try:
            with self._lock:
                if session_id not in self.user_ai_history:
                    self.user_ai_history[session_id] = []
                
                message = {
                    "role": role,
                    "content": content,
                    "timestamp": datetime.now().isoformat(),
                    "agent": agent if role == "assistant" else None
                }
                
                self.user_ai_history[session_id].append(message)
                
                # REQUIRED: Limit history length
                if len(self.user_ai_history[session_id]) > 100:
                    self.user_ai_history[session_id] = self.user_ai_history[session_id][-50:]
                
            logger.debug(f"Added user-AI message for session {session_id}: {role}")
            
        except Exception as e:
            logger.error(f"Failed to add user-AI message: {e}")
    
    def add_ai_ai_message(self, session_id: str, role: str, content: str, agent: str) -> None:
        """
        Add message to AI-AI coordination stream.
        
        Args:
            session_id: User session identifier
            role: Message role
            content: Message content
            agent: AI agent name
        """
        try:
            with self._lock:
                if session_id not in self.ai_ai_history:
                    self.ai_ai_history[session_id] = []
                
                message = {
                    "role": role,
                    "content": content,
                    "agent": agent,
                    "timestamp": datetime.now().isoformat()
                }
                
                self.ai_ai_history[session_id].append(message)
                
            logger.debug(f"Added AI-AI message for session {session_id}: {agent}")
            
        except Exception as e:
            logger.error(f"Failed to add AI-AI message: {e}")
    
    def get_user_ai_history(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get user-AI conversation history.
        
        Args:
            session_id: User session identifier
            limit: Maximum number of messages to return
            
        Returns:
            List of conversation messages
        """
        try:
            with self._lock:
                history = self.user_ai_history.get(session_id, [])
                
                if limit:
                    return history[-limit:]
                return history
                
        except Exception as e:
            logger.error(f"Failed to get user-AI history: {e}")
            return []
    
    def get_ai_ai_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get AI-AI coordination history.
        
        Args:
            session_id: User session identifier
            
        Returns:
            List of AI coordination messages
        """
        try:
            with self._lock:
                return self.ai_ai_history.get(session_id, [])
                
        except Exception as e:
            logger.error(f"Failed to get AI-AI history: {e}")
            return []
    
    def reset_ai_conversation(self, session_id: str) -> None:
        """
        Reset AI-AI conversation after final response.
        
        Args:
            session_id: User session identifier
        """
        try:
            with self._lock:
                if session_id in self.ai_ai_history:
                    self.ai_ai_history[session_id] = []
                    
            logger.debug(f"Reset AI-AI conversation for session {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to reset AI conversation: {e}")
    
    def cache_context(self, session_id: str, context_type: str, context_data: Any) -> None:
        """
        Cache context data for session.
        
        Args:
            session_id: User session identifier
            context_type: Type of context (e.g., 'news', 'alerts', 'portfolio')
            context_data: Context data to cache
        """
        try:
            with self._lock:
                if session_id not in self.context_cache:
                    self.context_cache[session_id] = {}
                
                self.context_cache[session_id][context_type] = {
                    "data": context_data,
                    "timestamp": datetime.now().isoformat()
                }
                
            logger.debug(f"Cached {context_type} context for session {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to cache context: {e}")
    
    def get_cached_context(self, session_id: str, context_type: str) -> Optional[Any]:
        """
        Get cached context data.
        
        Args:
            session_id: User session identifier
            context_type: Type of context to retrieve
            
        Returns:
            Cached context data or None
        """
        try:
            with self._lock:
                session_cache = self.context_cache.get(session_id, {})
                context_entry = session_cache.get(context_type)
                
                if context_entry:
                    return context_entry["data"]
                    
                return None
                
        except Exception as e:
            logger.error(f"Failed to get cached context: {e}")
            return None
    
    def set_session_metadata(self, session_id: str, metadata: Dict[str, Any]) -> None:
        """
        Set session metadata.
        
        Args:
            session_id: User session identifier
            metadata: Session metadata dictionary
        """
        try:
            with self._lock:
                self.session_metadata[session_id] = {
                    **metadata,
                    "created_at": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat()
                }
                
            logger.debug(f"Set metadata for session {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to set session metadata: {e}")
    
    def get_session_metadata(self, session_id: str) -> Dict[str, Any]:
        """
        Get session metadata.
        
        Args:
            session_id: User session identifier
            
        Returns:
            Session metadata dictionary
        """
        try:
            with self._lock:
                return self.session_metadata.get(session_id, {})
                
        except Exception as e:
            logger.error(f"Failed to get session metadata: {e}")
            return {}
    
    def update_session_metadata(self, session_id: str, updates: Dict[str, Any]) -> None:
        """
        Update session metadata.
        
        Args:
            session_id: User session identifier
            updates: Updates to apply to metadata
        """
        try:
            with self._lock:
                if session_id in self.session_metadata:
                    self.session_metadata[session_id].update(updates)
                    self.session_metadata[session_id]["last_updated"] = datetime.now().isoformat()
                else:
                    self.set_session_metadata(session_id, updates)
                    
            logger.debug(f"Updated metadata for session {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to update session metadata: {e}")
    
    def clear_session(self, session_id: str) -> None:
        """
        Clear all data for a session.
        
        Args:
            session_id: User session identifier
        """
        try:
            with self._lock:
                # REQUIRED: Clear all session data
                if session_id in self.user_ai_history:
                    del self.user_ai_history[session_id]
                
                if session_id in self.ai_ai_history:
                    del self.ai_ai_history[session_id]
                
                if session_id in self.context_cache:
                    del self.context_cache[session_id]
                
                if session_id in self.session_metadata:
                    del self.session_metadata[session_id]
                    
            logger.info(f"Cleared all data for session {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to clear session: {e}")
    
    def get_conversation_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Get conversation summary for session.
        
        Args:
            session_id: User session identifier
            
        Returns:
            Conversation summary with statistics
        """
        try:
            with self._lock:
                user_ai_messages = self.user_ai_history.get(session_id, [])
                ai_ai_messages = self.ai_ai_history.get(session_id, [])
                
                # REQUIRED: Count messages by role
                user_message_count = len([m for m in user_ai_messages if m["role"] == "user"])
                assistant_message_count = len([m for m in user_ai_messages if m["role"] == "assistant"])
                
                # REQUIRED: Get unique agents used
                agents_used = list(set([m.get("agent") for m in user_ai_messages if m.get("agent")]))
                
                return {
                    "session_id": session_id,
                    "user_messages": user_message_count,
                    "assistant_messages": assistant_message_count,
                    "total_user_ai_messages": len(user_ai_messages),
                    "total_ai_ai_messages": len(ai_ai_messages),
                    "agents_used": agents_used,
                    "session_start": user_ai_messages[0]["timestamp"] if user_ai_messages else None,
                    "last_activity": user_ai_messages[-1]["timestamp"] if user_ai_messages else None
                }
                
        except Exception as e:
            logger.error(f"Failed to get conversation summary: {e}")
            return {}
    
    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """
        Clean up old sessions.
        
        Args:
            max_age_hours: Maximum age of sessions to keep
            
        Returns:
            Number of sessions cleaned up
        """
        try:
            from datetime import timedelta
            
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            sessions_to_remove = []
            
            with self._lock:
                for session_id in list(self.session_metadata.keys()):
                    metadata = self.session_metadata[session_id]
                    last_updated = datetime.fromisoformat(metadata.get("last_updated", ""))
                    
                    if last_updated < cutoff_time:
                        sessions_to_remove.append(session_id)
                
                # REQUIRED: Remove old sessions
                for session_id in sessions_to_remove:
                    self.clear_session(session_id)
                    
            logger.info(f"Cleaned up {len(sessions_to_remove)} old sessions")
            return len(sessions_to_remove)
            
        except Exception as e:
            logger.error(f"Failed to cleanup old sessions: {e}")
            return 0
    
    def get_memory_usage(self) -> Dict[str, int]:
        """
        Get memory usage statistics.
        
        Returns:
            Dictionary with memory usage stats
        """
        try:
            with self._lock:
                stats = {
                    "total_sessions": len(self.session_metadata),
                    "user_ai_sessions": len(self.user_ai_history),
                    "ai_ai_sessions": len(self.ai_ai_history),
                    "cached_contexts": sum(len(cache) for cache in self.context_cache.values()),
                    "total_user_ai_messages": sum(len(history) for history in self.user_ai_history.values()),
                    "total_ai_ai_messages": sum(len(history) for history in self.ai_ai_history.values())
                }
                
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get memory usage: {e}")
            return {}
    
    def export_session_data(self, session_id: str) -> Dict[str, Any]:
        """
        Export all data for a session.
        
        Args:
            session_id: User session identifier
            
        Returns:
            Complete session data
        """
        try:
            with self._lock:
                return {
                    "session_id": session_id,
                    "metadata": self.session_metadata.get(session_id, {}),
                    "user_ai_history": self.user_ai_history.get(session_id, []),
                    "ai_ai_history": self.ai_ai_history.get(session_id, []),
                    "context_cache": self.context_cache.get(session_id, {}),
                    "exported_at": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Failed to export session data: {e}")
            return {}
    
    def import_session_data(self, session_data: Dict[str, Any]) -> bool:
        """
        Import session data.
        
        Args:
            session_data: Complete session data to import
            
        Returns:
            True if import successful
        """
        try:
            session_id = session_data.get("session_id")
            if not session_id:
                logger.error("No session_id in import data")
                return False
            
            with self._lock:
                # REQUIRED: Import all data streams
                if "metadata" in session_data:
                    self.session_metadata[session_id] = session_data["metadata"]
                
                if "user_ai_history" in session_data:
                    self.user_ai_history[session_id] = session_data["user_ai_history"]
                
                if "ai_ai_history" in session_data:
                    self.ai_ai_history[session_id] = session_data["ai_ai_history"]
                
                if "context_cache" in session_data:
                    self.context_cache[session_id] = session_data["context_cache"]
                    
            logger.info(f"Imported session data for {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import session data: {e}")
            return False 