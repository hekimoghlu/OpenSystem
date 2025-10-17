/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 25, 2021.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
//
// sqlite++ - C++ interface to SQLite3
//
#ifndef _H_SQLITEPP
#define _H_SQLITEPP

#include <sqlite3.h>
#include <security_utilities/errors.h>
#include <security_utilities/threading.h>
#include <CoreFoundation/CFData.h>


namespace Security {
namespace SQLite3 {

class Database;
class Statement;

typedef sqlite3_int64 int64;
typedef sqlite3_uint64 uint64;


//
// An sqlite3 error
//
class Error : public CommonError {
public:
	Error(Database &db);
	Error(int err) : error(err) { }
	Error(int err, const char *msg) : error(err), message(msg) { }
	~Error() _NOEXCEPT { }
	const int error;
	const std::string message;
	
	const char *what() const _NOEXCEPT { return message.c_str(); }
    OSStatus osStatus() const;
	int unixError() const;
	
	static void check(int err);
	static void throwMe(int err) __attribute__((noreturn));
};


//
// An sqlite3 database "connection"
//
class Database {
	friend class Statement;
public:
	Database(const char *path, int flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, bool lenient = false);
	virtual ~Database();
	
	bool isOpen() const { return mDb != NULL; }
	void close();
	
	// open flags
	int openFlags() const { return mOpenFlags; }
	
	// last error condition encountered
	int errcode();
	const char *errmsg();
	
	bool inTransaction();
	int64 lastInsert();
	int changes();
	
	void interrupt();
	
	int execute(const char *text, bool strict = true);
	int execute(const std::string &text, bool strict = true)
		{ return execute(text.c_str(), strict); }
	
	bool empty();
	
	template <class RType> RType value(const char *text, RType defaultResult = RType());
	template <class RType> RType value(const std::string &text, RType defaultResult = RType())
		{ return value(text.c_str(), defaultResult); }
	
	double julianNow()
		{ return this->value<double>("SELECT JULIANDAY('now');"); }
	
	void busyDelay(int ms);

	void check(int err);
	
	sqlite3 *sql() const { return mDb; }

private:
	sqlite3 *mDb;
	Mutex mMutex;
	int mOpenFlags;
};


//
// An sqlite column value.
// These are definitely not first-class API objects; in particular,
// there doesn't seem to be API to actually *make* one - you can only
// get them out of sqlite.
//
class Value {
public:
	Value(sqlite3_value *v) : mValue(v) { }

	operator int () const { return ::sqlite3_value_int(mValue); }
	operator sqlite3_int64 () const { return ::sqlite3_value_int64(mValue); }
	operator const char * () const { return (const char *)::sqlite3_value_text(mValue); }
	operator double () const { return ::sqlite3_value_double(mValue); }
	
	int type() const { return ::sqlite3_value_type(mValue); }
	int numericType() const { return ::sqlite3_value_numeric_type(mValue); }
	
	operator bool () const { return type() != SQLITE_NULL; }
	bool operator ! () const { return type() == SQLITE_NULL; }
	
	sqlite3_value *sql() const { return mValue; }

private:
	sqlite3_value *mValue;
};


//
// A Transaction proxy.
//
class Transaction {
public:	
	enum Type {
		deferred,
		immediate,
		exclusive
	};

public:
	Transaction(Database &db, Type type = deferred, const char *name = NULL);
	virtual ~Transaction();
	
	void commit();
	void abort();
	void rollback() { this->abort(); }
	
	Database &database;

protected:
	void xactCommand(const std::string &s);

private:
	std::string mName;
};


//
// A (prepared) statement.
//
class Statement : private StLock<Mutex> {
	class Binding;
	
public:
	Statement(Database &db, const char *text);	// ready to serve
	Statement(Database &db);						// quiescent; call query(text) to activate it
	virtual ~Statement();
	
	Database &database;

	operator bool () const { return mStmt != NULL; } // active
	
	void query(const char *text);					// activate statement with query text
	void query(const std::string &text)
		{ query(text.c_str()); }
	void close();									// close up active statement

	Binding bind(int ix) const { return Binding(*this, ix); }
	Binding bind(const char *name) const;
	unsigned int bindings() const { return ::sqlite3_bind_parameter_count(mStmt); }
	void unbind();

	int step();
	void execute();
	bool nextRow();
	bool operator () () { return nextRow(); }

	void reset();
	
	class Result;
	Result operator [] (int ix) { return Result(*this, ix); }
	unsigned int count() const { return ::sqlite3_column_count(mStmt); }
	
	void check(int err) const { database.check(err); }
	sqlite3_stmt *sql() const { return mStmt; }

private:
	class Column {
	public:
		Column(const Statement &st, int ix) : statement(st), index(ix) { }
		
		const Statement &statement;
		const int index;
	};
	
	class Binding : public Column {
	public:
		Binding(const Statement &st, int ix) : Column(st, ix) { }
		
		const char *name() const;
		
		void null();
		void operator = (int value);
		void operator = (sqlite3_int64 value);
		void operator = (double value);
		void operator = (const char *value);
		void operator = (const std::string &value);
		void operator = (const Value &value);
		void integer(sqlite3_int64 value);
		void blob(const void *data, size_t length, bool shared = false);
		void operator = (CFDataRef data);
		void operator = (CFStringRef value);
	};
	
public:
	class Result : public Column {
	public:
		Result(const Statement &st, int ix) : Column(st, ix) { }
		
		const char *name() const;
		
		operator int () const { return ::sqlite3_column_int(statement.sql(), index); }
		operator sqlite3_int64 () const { return ::sqlite3_column_int64(statement.sql(), index); }
		operator double () const { return ::sqlite3_column_double(statement.sql(), index); }
		const char *string() const { return (const char *)::sqlite3_column_text(statement.sql(), index); }
		operator const char *() const { return this->string(); }
		const void *blob() const { return ::sqlite3_column_blob(statement.sql(), index); }
		int length() const { return ::sqlite3_column_bytes(statement.sql(), index); }
		CFDataRef data() const;
		
		int type() const { return ::sqlite3_column_type(statement.sql(), index); }
		const char *declType() const { return ::sqlite3_column_decltype(statement.sql(), index); }
	
		operator bool () const { return type() != SQLITE_NULL; }
		bool operator ! () const { return type() == SQLITE_NULL; }
	};

private:
	sqlite3_stmt *mStmt;
};


template <class RType>
RType Database::value(const char *text, RType defaultResult)
{
	Statement stmt(*this, text);
	if (stmt())
		return RType(stmt[0]);
	else
		return defaultResult;
}



} // SQLite3
}	// Security

#endif //_H_SQLITEPP
