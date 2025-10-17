/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 22, 2022.
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
#include <iostream>

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <uuid/uuid.h>
#include <errno.h>
#include <_simple.h>
#include <unistd.h>
#include <sys/uio.h>
#include <mach/mach.h>
#include <mach/machine.h>
#include <mach-o/fat.h>
#include <libc_private.h>

#include "Error.h"

namespace error {


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

Error::~Error()
{
   if ( _buffer )
        _simple_sfree(_buffer);
    _buffer = nullptr;
}

Error::Error(const char* format, ...)
{
    va_list    list;
    va_start(list, format);
    if ( _buffer == nullptr )
        _buffer = _simple_salloc();
    _simple_vsprintf(_buffer, format, list);
    va_end(list);
}

const char* Error::message() const
{
    return _buffer ? _simple_string(_buffer) : "";
}

bool Error::messageContains(const char* subString) const
{
    if ( _buffer == nullptr )
        return false;
    return (strstr(_simple_string(_buffer), subString) != nullptr);
}

} // namespace error
