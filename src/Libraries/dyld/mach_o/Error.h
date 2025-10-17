/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 22, 2023.
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
#ifndef mach_o_Error_h
#define mach_o_Error_h

#include <TargetConditionals.h>
#include <stdarg.h>

#include "MachODefines.h"
#include "va_list_wrap.h"

namespace mach_o {

/*!
 * @class Error
 *
 * @abstract
 *      Class for capturing error messages.
 *      Can be constructed with printf style format strings.
 *      Returned by mach-o "valid" methods. 
 */
class VIS_HIDDEN [[nodiscard]] Error
{
public:
                    Error() = default;
                    Error(const char* format, ...)  __attribute__((format(printf, 2, 3)));
                    Error(const char* format, va_list_wrap vaWrap) __attribute__((format(printf, 2, 0)));
                    Error(Error&&); // can move
                    Error& operator=(const Error&) = delete; //  can't copy assign
                    Error& operator=(Error&&); // can move
                    ~Error();


    void            append(const char* format, ...)  __attribute__((format(printf, 2, 3)));
    bool            hasError() const { return (_buffer != nullptr); }
    bool            noError() const  { return (_buffer == nullptr); }
    explicit        operator bool() const { return hasError(); }
    const char*     message() const;
    bool            messageContains(const char* subString) const;

    static Error    copy(const Error&);

    static Error    none() { return Error(); }

private:

#if TARGET_OS_EXCLAVEKIT
    char            _strBuf[1024];
#endif
    void*           _buffer = nullptr;
};



} // namespace mach_o

#endif /* mach_o_Error_h */
