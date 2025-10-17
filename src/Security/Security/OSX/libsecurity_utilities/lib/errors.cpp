/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 12, 2023.
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
#include <security_utilities/errors.h>
#include <security_utilities/debugging.h>
#include <security_utilities/utility_config.h>
#include <security_utilities/debugging_internal.h>
#include <typeinfo>
#include <stdio.h>
#include <Security/SecBase.h>
#include <Security/CSCommon.h>
#include <execinfo.h>
#include <cxxabi.h>

//@@@
// From cssmapple.h - layering break
// Where should this go?
//@@@
#define errSecErrnoBase 100000
#define errSecErrnoLimit 100255

//
// The base of the exception hierarchy.
//
CommonError::CommonError() : whatBuffer("CommonError")
{
}


//
// We strongly encourage catching all exceptions by const reference, so the copy
// constructor of our exceptions should never be called.
//
CommonError::CommonError(const CommonError &source)
{
    strlcpy(whatBuffer, source.whatBuffer, whatBufferSize);
}

CommonError::~CommonError() _NOEXCEPT
{
}

void CommonError::LogBacktrace() {
    // Only do this work if we're actually going to log things
    if(secinfoenabled("security_exception")) {
        const size_t maxsize = 32;
        void* callstack[maxsize];

        int size = backtrace(callstack, maxsize);
        char** names = backtrace_symbols(callstack, size);

        // C++ symbolicate the callstack

        const char* delim = " ";
        string build;
        char * token = NULL;
        char * line = NULL;

        for(int i = 0; i < size; i++) {
            build = "";

            line = names[i];

            while((token = strsep(&line, delim))) {
                if(*token == '\0') {
                    build += " ";
                } else {
                    int status = 0;
                    char * demangled = abi::__cxa_demangle(token, NULL, NULL, &status);
                    if(status == 0) {
                        build += demangled;
                    } else {
                        build += token;
                    }
                    build += " ";

                    if(demangled) {
                        free(demangled);
                    }
                }
            }

            secinfo("security_exception", "%s", build.c_str());
        }
        free(names);
    }
}



//
// UnixError exceptions
//
UnixError::UnixError() : error(errno)
{
    SECURITY_EXCEPTION_THROW_UNIX(this, errno);

    snprintf(whatBuffer, whatBufferSize, "UNIX errno exception: %d", this->error);
    secnotice("security_exception", "%s", what());
    LogBacktrace();
}

UnixError::UnixError(int err, bool suppresslogging) : error(err)
{
    SECURITY_EXCEPTION_THROW_UNIX(this, err);

    if(!suppresslogging || secinfoenabled("security_exception")) {
        snprintf(whatBuffer, whatBufferSize, "UNIX error exception: %d", this->error);
        secnotice("security_exception", "%s", what());
        LogBacktrace();
    }
}

const char *UnixError::what() const _NOEXCEPT
{
    return whatBuffer;
}


OSStatus UnixError::osStatus() const
{
	return error + errSecErrnoBase;
}

int UnixError::unixError() const
{ return error; }

void UnixError::throwMe(int err) { throw UnixError(err, false); }
void UnixError::throwMeNoLogging(int err) { throw UnixError(err, true); }


// @@@ This is a hack for the Network protocol state machine
UnixError UnixError::make(int err) { return UnixError(err, false); }


//
// MacOSError exceptions
//
MacOSError::MacOSError(int err) : error(err)
{
    SECURITY_EXCEPTION_THROW_OSSTATUS(this, err);

    snprintf(whatBuffer, whatBufferSize, "MacOS error: %d", this->error);
    switch (err) {
        case errSecCSReqFailed:
            // This 'error' isn't an actual error and doesn't warrant being logged.
            break;
        default:
            secnotice("security_exception", "%s", what());
            LogBacktrace();
    }
}

const char *MacOSError::what() const _NOEXCEPT
{
    return whatBuffer;
}

OSStatus MacOSError::osStatus() const
{ return error; }

int MacOSError::unixError() const
{
	// embedded UNIX errno values are returned verbatim
	if (error >= errSecErrnoBase && error <= errSecErrnoLimit)
		return error - errSecErrnoBase;

	switch (error) {
	default:
		// cannot map this to errno space
		return -1;
    }
}

void MacOSError::throwMe(int error)
{ throw MacOSError(error); }

void MacOSError::throwMe(int error, char const *message, ...)
{
    // Ignoring the message for now, will do something with it later.
    throw MacOSError(error);
}

MacOSError MacOSError::make(int error)
{ return MacOSError(error); }


//
// CFError exceptions
//
CFError::CFError()
{
    SECURITY_EXCEPTION_THROW_CF(this);
    secnotice("security_exception", "CFError");
    LogBacktrace();
}

const char *CFError::what() const _NOEXCEPT
{ return "CoreFoundation error"; }

OSStatus CFError::osStatus() const
{ return errSecCoreFoundationUnknown; }

int CFError::unixError() const
{
	return EFAULT;		// nothing really matches
}

void CFError::throwMe()
{ throw CFError(); }




void ModuleNexusError::throwMe()
{
    throw ModuleNexusError();
}



OSStatus ModuleNexusError::osStatus() const
{
    return errSecParam;
}



int ModuleNexusError::unixError() const
{
    return EINVAL;      
}

