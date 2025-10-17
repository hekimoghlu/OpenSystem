/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 24, 2023.
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
// debugging_internal - non-trivial debug support
//
// everything in this file is deprecated. Try not to use it.
//
#ifndef _H_DEBUGGING
#define _H_DEBUGGING

#ifdef __cplusplus

#include <security_utilities/utilities.h>
#include <cstdarg>
#include <typeinfo>

namespace Security {
namespace Debug {


//
// Debug-dumping functions always exist. They may be stubs depending on build options.
//
bool dumping(const char *scope);
void dump(const char *format, ...) __attribute((format(printf,1,2)));
void dumpData(const void *data, size_t length);
void dumpData(const char *title, const void *data, size_t length);
template <class Data> inline void dumpData(const Data &obj)
{ dumpData(obj.data(), obj.length()); }
template <class Data> inline void dumpData(const char *title, const Data &obj) 
{ dumpData(title, obj.data(), obj.length()); }


//
// The following functions perform runtime recovery of type names.
// This is meant for debugging ONLY. Don't even THINK of depending
// on this for program correctness. For all you know, we may replace
// all those names with "XXX" tomorrow.
//
string makeTypeName(const type_info &info);

template <class Object>
string typeName(const Object &obj)
{
	return makeTypeName(typeid(obj));
}

template <class Object>
string typeName()
{
	return makeTypeName(typeid(Object));
}


//
// We are still conditionally emitting debug-dumping code
//
#undef DEBUGGING
#if !defined(NDEBUG)
# define DEBUGGING 1
// No more debugdump, it emits thread-unsafe buggy code which hampers actual debugging, ironically enough
// # define DEBUGDUMP 1
#else //NDEBUG
# define DEBUGGING 0
#endif //NDEBUG

#if defined(DEBUGDUMP)
# define IFDUMP(code)				code
# define IFDUMPING(scope,code)		if (Debug::dumping(scope)) code; else /* no */
#else
# define IFDUMP(code)				/* no-op */
# define IFDUMPING(scope,code)		/* no-op */
#endif


//
// We have some very, very old customers who call old debug facilities.
// Dummy them out for now.
//
inline bool debugging(const char *scope) DEPRECATED_ATTRIBUTE;
inline void debug(const char *scope, const char *format, ...) DEPRECATED_ATTRIBUTE;
inline void vdebug(const char *scope, const char *format, va_list args) DEPRECATED_ATTRIBUTE;

inline bool debugging(const char *scope) { return false; }
inline void debug(const char *scope, const char *format, ...) { }
inline void vdebug(const char *scope, const char *format, va_list args) { }





} // end namespace Debug
} // end namespace Security

// leak debug() into the global namespace because URLAccess et al rely on that
using Security::Debug::debug;

__BEGIN_DECLS

//
// Include DTrace static probe definitions
//
typedef const void *DTException;
#include <security_utilities/utilities_dtrace.h>

// The following are deprecated functions. Don't use them (but they need to be here for symbol reasons).
__attribute__((visibility("default"))) void secdebug_internal(const char* scope, const char* format, ...);
__attribute__((visibility("default"))) void secdebugfunc_internal(const char* scope, const char* functionname, const char* format, ...);

__END_DECLS

#else	//__cplusplus

#include <stdio.h>

#endif	//__cplusplus

#include <CoreFoundation/CFString.h>


#endif //_H_DEBUGGING
