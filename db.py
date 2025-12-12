import os
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING, DESCENDING
from bson.objectid import ObjectId
import pytz

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "flood_forecasting_db")

client = MongoClient(MONGODB_URI)
db = client[DB_NAME]

# Collections
users_collection = db.users
activity_logs_collection = db.activity_logs
forecasts_collection = db.forecasts
searches_collection = db.searches

# Indian Standard Time helper - FIXED VERSION
def get_ist_time():
    """Get current time in IST (Asia/Kolkata timezone)"""
    ist = pytz.timezone('Asia/Kolkata')
    return datetime.now(ist)

# Create indexes for better performance
def init_db():
    """Initialize database with proper indexes"""
    try:
        # User indexes
        users_collection.create_index([("username", ASCENDING)], unique=True)
        users_collection.create_index([("email", ASCENDING)], unique=True)
        
        # Activity logs indexes
        activity_logs_collection.create_index([("username", ASCENDING)])
        activity_logs_collection.create_index([("timestamp", DESCENDING)])
        activity_logs_collection.create_index([("activity_type", ASCENDING)])
        
        # Forecasts indexes
        forecasts_collection.create_index([("username", ASCENDING)])
        forecasts_collection.create_index([("state", ASCENDING)])
        forecasts_collection.create_index([("district", ASCENDING)])
        forecasts_collection.create_index([("created_at", DESCENDING)])
        
        # Searches indexes
        searches_collection.create_index([("username", ASCENDING)])
        searches_collection.create_index([("timestamp", DESCENDING)])
        searches_collection.create_index([("state", ASCENDING)])
        searches_collection.create_index([("district", ASCENDING)])
        
        print("✅ Database indexes created successfully")
        return True
    except Exception as e:
        print(f"⚠️ Note: Some indexes may already exist - {e}")
        return False

# User Management Functions
def create_user(username, password_hash, email, name):
    """Create a new user"""
    try:
        user_data = {
            "username": username,
            "password": password_hash,
            "email": email,
            "name": name,
            "created_at": get_ist_time(),
            "last_login": None,
            "login_count": 0,
            "is_active": True
        }
        result = users_collection.insert_one(user_data)
        print(f"✅ User created: {username}")
        return str(result.inserted_id)
    except Exception as e:
        print(f"❌ Error creating user: {e}")
        return None

def get_user(username):
    """Get user by username"""
    return users_collection.find_one({"username": username})

def get_all_users():
    """Get all users as dictionary"""
    users = {}
    for user in users_collection.find():
        users[user["username"]] = {
            "password": user["password"],
            "email": user["email"],
            "name": user["name"]
        }
    return users

def update_user_login(username):
    """Update user login timestamp and count"""
    users_collection.update_one(
        {"username": username},
        {
            "$set": {"last_login": get_ist_time()},
            "$inc": {"login_count": 1}
        }
    )
    print(f"✅ Login updated for: {username}")

def user_exists(username):
    """Check if user exists"""
    return users_collection.count_documents({"username": username}) > 0

def email_exists(email):
    """Check if email exists"""
    return users_collection.count_documents({"email": email}) > 0

# Activity Logging Functions
def log_activity(username, activity_type, details=None):
    """Log user activity"""
    try:
        activity_data = {
            "username": username,
            "activity_type": activity_type,
            "details": details or {},
            "timestamp": get_ist_time()
        }
        result = activity_logs_collection.insert_one(activity_data)
        print(f"✅ Activity logged: {username} - {activity_type}")
        return str(result.inserted_id)
    except Exception as e:
        print(f"❌ Error logging activity: {e}")
        return None

def get_user_activities(username, limit=50):
    """Get user's recent activities"""
    return list(activity_logs_collection.find(
        {"username": username}
    ).sort("timestamp", DESCENDING).limit(limit))

def get_all_activities(limit=100):
    """Get all recent activities"""
    return list(activity_logs_collection.find().sort("timestamp", DESCENDING).limit(limit))

# Search Tracking Functions
def log_search(username, state, district, search_type="location_search"):
    """Log user search"""
    try:
        search_data = {
            "username": username,
            "state": state,
            "district": district,
            "search_type": search_type,
            "timestamp": get_ist_time()
        }
        result = searches_collection.insert_one(search_data)
        
        # Also log as activity
        log_activity(username, "search", {
            "state": state,
            "district": district,
            "search_type": search_type
        })
        
        print(f"✅ Search logged: {username} - {district}, {state}")
        return str(result.inserted_id)
    except Exception as e:
        print(f"❌ Error logging search: {e}")
        return None

def get_user_searches(username, limit=20):
    """Get user's recent searches"""
    return list(searches_collection.find(
        {"username": username}
    ).sort("timestamp", DESCENDING).limit(limit))

def get_popular_locations(limit=10):
    """Get most searched locations"""
    pipeline = [
        {"$group": {
            "_id": {"state": "$state", "district": "$district"},
            "count": {"$sum": 1}
        }},
        {"$sort": {"count": DESCENDING}},
        {"$limit": limit}
    ]
    return list(searches_collection.aggregate(pipeline))

# Forecast Management Functions
def save_forecast(username, state, district, forecast_data, forecast_year, forecast_months):
    """Save generated forecast"""
    try:
        forecast_document = {
            "username": username,
            "state": state,
            "district": district,
            "forecast_year": forecast_year,
            "forecast_months": forecast_months,
            "forecast_data": forecast_data,
            "metadata": {
                "total_rainfall": sum(forecast_data) if forecast_data else 0,
                "avg_rainfall": sum(forecast_data) / len(forecast_data) if forecast_data else 0,
                "max_rainfall": max(forecast_data) if forecast_data else 0,
                "min_rainfall": min(forecast_data) if forecast_data else 0,
            },
            "created_at": get_ist_time()
        }
        result = forecasts_collection.insert_one(forecast_document)
        
        # Log activity
        log_activity(username, "forecast_generated", {
            "state": state,
            "district": district,
            "forecast_year": forecast_year,
            "forecast_months": forecast_months,
            "forecast_id": str(result.inserted_id)
        })
        
        print(f"✅ Forecast saved: {username} - {district}, {state}")
        return str(result.inserted_id)
    except Exception as e:
        print(f"❌ Error saving forecast: {e}")
        return None

def get_user_forecasts(username, limit=10):
    """Get user's recent forecasts"""
    return list(forecasts_collection.find(
        {"username": username}
    ).sort("created_at", DESCENDING).limit(limit))

def get_forecast_by_id(forecast_id):
    """Get specific forecast by ID"""
    try:
        return forecasts_collection.find_one({"_id": ObjectId(forecast_id)})
    except:
        return None

def delete_forecast(forecast_id):
    """Delete a forecast"""
    try:
        result = forecasts_collection.delete_one({"_id": ObjectId(forecast_id)})
        if result.deleted_count > 0:
            print(f"✅ Forecast deleted: {forecast_id}")
            return True
        return False
    except Exception as e:
        print(f"❌ Error deleting forecast: {e}")
        return False

# Analytics Functions
def get_user_statistics(username):
    """Get user statistics"""
    try:
        user = get_user(username)
        if not user:
            return None
        
        total_searches = searches_collection.count_documents({"username": username})
        total_forecasts = forecasts_collection.count_documents({"username": username})
        total_activities = activity_logs_collection.count_documents({"username": username})
        
        # Get most searched location
        pipeline = [
            {"$match": {"username": username}},
            {"$group": {
                "_id": {"state": "$state", "district": "$district"},
                "count": {"$sum": 1}
            }},
            {"$sort": {"count": DESCENDING}},
            {"$limit": 1}
        ]
        most_searched = list(searches_collection.aggregate(pipeline))
        
        return {
            "username": username,
            "name": user.get("name"),
            "email": user.get("email"),
            "member_since": user.get("created_at"),
            "last_login": user.get("last_login"),
            "login_count": user.get("login_count", 0),
            "total_searches": total_searches,
            "total_forecasts": total_forecasts,
            "total_activities": total_activities,
            "most_searched_location": most_searched[0]["_id"] if most_searched else None
        }
    except Exception as e:
        print(f"❌ Error getting user statistics: {e}")
        return None

def get_system_statistics():
    """Get overall system statistics"""
    try:
        total_users = users_collection.count_documents({})
        total_searches = searches_collection.count_documents({})
        total_forecasts = forecasts_collection.count_documents({})
        total_activities = activity_logs_collection.count_documents({})
        
        # Active users (logged in last 7 days)
        week_ago = get_ist_time() - timedelta(days=7)
        active_users = users_collection.count_documents({"last_login": {"$gte": week_ago}})
        
        return {
            "total_users": total_users,
            "active_users": active_users,
            "total_searches": total_searches,
            "total_forecasts": total_forecasts,
            "total_activities": total_activities
        }
    except Exception as e:
        print(f"❌ Error getting system statistics: {e}")
        return None

# Download Tracking
def log_download(username, download_type, state, district, forecast_id=None):
    """Log file downloads"""
    try:
        log_activity(username, "download", {
            "download_type": download_type,
            "state": state,
            "district": district,
            "forecast_id": forecast_id
        })
        print(f"✅ Download logged: {username} - {download_type}")
        return True
    except Exception as e:
        print(f"❌ Error logging download: {e}")
        return False

# Test connection
def test_connection():
    """Test MongoDB connection"""
    try:
        client.server_info()
        print("✅ MongoDB connection successful!")
        print(f"📊 Database: {db.name}")
        print(f"🕐 Current IST Time: {get_ist_time().strftime('%Y-%m-%d %I:%M:%S %p IST')}")
        return True
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        return False

# Initialize database on import
init_db()

# Test connection
if __name__ != "__main__":
    test_connection()