/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 29, 2025.
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
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <uuid/uuid.h>
#include <errno.h>
#include <TargetConditionals.h>
#if !TARGET_OS_EXCLAVEKIT
  #include <_simple.h>
#endif // !TARGET_OS_EXCLAVEKIT

#include <mach/machine.h>
#include <mach-o/fat.h>

#include "Error.h"

namespace mach_o {


Error::Error(Error&& other)
{
    _buffer = other._buffer;
    other._buffer = nullptr;
}

Error& Error::operator=(Error&& other)
{
    _buffer = other._buffer;
    other._buffer = nullptr;
    return *this;
}


Error Error::copy(const Error& other)
{
    if ( other.noError() )
        return Error::none();
    return Error("%s", other.message());
}

Error::~Error()
{
#if TARGET_OS_EXCLAVEKIT
    *_strBuf = '\0';
#else
   if ( _buffer )
        _simple_sfree(_buffer);
    _buffer = nullptr;
#endif
}

Error::Error(const char* format, ...)
{
    va_list    list;
    va_start(list, format);
#if TARGET_OS_EXCLAVEKIT
    vsnprintf(_strBuf, sizeof(_strBuf), format, list);
#else
    if ( _buffer == nullptr )
        _buffer = _simple_salloc();
    _simple_vsprintf(_buffer, format, list);
#endif
    va_end(list);
}

Error::Error(const char* format, va_list_wrap vaWrap)
{
#if TARGET_OS_EXCLAVEKIT
    vsnprintf(_strBuf, sizeof(_strBuf), format, vaWrap.list);
#else
    if ( _buffer == nullptr )
        _buffer = _simple_salloc();
    _simple_vsprintf(_buffer, format, vaWrap.list);
#endif
}

const char* Error::message() const
{
#if TARGET_OS_EXCLAVEKIT
    return _strBuf;
#else
    return _buffer ? _simple_string(_buffer) : "";
#endif
}

bool Error::messageContains(const char* subString) const
{
    if ( _buffer == nullptr )
        return false;
#if TARGET_OS_EXCLAVEKIT
    return (strstr(_strBuf, subString) != nullptr);
#else
    return (strstr(_simple_string(_buffer), subString) != nullptr);
#endif
}

void Error::append(const char* format, ...)
{
#if TARGET_OS_EXCLAVEKIT
   size_t len = strlen(_strBuf);
   va_list list;
   va_start(list, format);
   vsnprintf(&_strBuf[len], sizeof(_strBuf)-len, format, list);
   va_end(list);
#else
    assert(_buffer != nullptr);
    _simple_sresize(_buffer);   // move insertion point to end of existing string in buffer
    va_list list;
    va_start(list, format);
    _simple_vsprintf(_buffer, format, list);
    va_end(list);
#endif
}


} // namespace mach_o
