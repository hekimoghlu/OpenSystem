/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 8, 2022.
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
// Error hierarchy
//
#ifndef _H_UTILITIES_ERROR
#define _H_UTILITIES_ERROR

#include <AvailabilityMacros.h>
#include <exception>
#include <errno.h>
#include <Security/SecBase.h>
#include <set>
#undef check


namespace Security {


//
// Common base of Security exceptions that represent error conditions.
// All can yield Unix or OSStatus error codes as needed, though *how*
// is up to the subclass implementation.
// CSSM_RETURN conversions are done externally in (???).
//
class CommonError : public std::exception {
protected:
    CommonError();
    CommonError(const CommonError &source);
public:
    virtual ~CommonError() _NOEXCEPT;

    virtual OSStatus osStatus() const = 0;
	virtual int unixError() const = 0;

    char whatBuffer[128];
    const size_t whatBufferSize = sizeof(whatBuffer);

    static void LogBacktrace();
};


//
// Genuine Unix-originated errors identified by an errno value.
// This includes secondary sources such as pthreads.
//
class UnixError : public CommonError {
protected:
    UnixError();
    UnixError(int err, bool suppresslogging);
public:
    const int error;
    virtual OSStatus osStatus() const;
	virtual int unixError() const;
    virtual const char *what () const _NOEXCEPT;
    
    static void check(int result)		{ if (result == -1) throwMe(); }
    static void throwMe(int err = errno) __attribute__((noreturn));
    static void throwMeNoLogging(int err = errno) __attribute__((noreturn));

    // @@@ This is a hack for the Network protocol state machine
    static UnixError make(int err = errno) DEPRECATED_ATTRIBUTE;
};


//
// Genuine MacOS (X) errors identified by an OSStatus value.
// Don't even think of working with OSErr values; use OSStatus.
//
class MacOSError : public CommonError {
protected:
    MacOSError(int err);
public:
    const int error;
    virtual OSStatus osStatus() const;
	virtual int unixError() const;
    virtual const char *what () const _NOEXCEPT;
    
    static void check(OSStatus status)	{ if (status != errSecSuccess) throwMe(status); }
    static void throwMe(int err) __attribute__((noreturn));
    static void throwMe(int err, char const *message, ...) __attribute__((noreturn));

    static MacOSError make(int err);
};
	
typedef std::set<OSStatus> MacOSErrorSet;


//
// CoreFoundation errors.
// Since CF prefers not to tell us *why* something didn't work, this
// is not very useful - but it's better than faking it into one of the other
// error spaces.
//
class CFError : public CommonError {
protected:
	CFError();
public:
	virtual OSStatus osStatus() const;
	virtual int unixError() const;
	virtual const char *what () const _NOEXCEPT;
	
	template <class T>
	static void check(const T &p)		{ if (!p) throwMe(); }

	static void throwMe() __attribute__((noreturn));
};


// Something that gets thrown when ModuleNexus creation fails
class ModuleNexusError : public CommonError {
protected:
    ModuleNexusError() {}

public:
    virtual OSStatus osStatus() const;
	virtual int unixError() const;
    static void throwMe() __attribute__((noreturn));
};

} // end namespace Security


#endif //_H_UTILITIES_ERROR
