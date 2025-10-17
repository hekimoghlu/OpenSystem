/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 26, 2022.
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
#define	__va_list	__darwin_va_list

#include <printf.h>
#include <stdarg.h>
#include <local.h>
#include <xprintf_private.h>

int
asxprintf(char ** __restrict ret, printf_domain_t __restrict domain,
    locale_t __restrict loc, const char * __restrict format, ...)
{
    int iret;
    va_list ap;

    va_start(ap, format);
    iret = _vasprintf(NULL, domain, ret, loc, format, ap);
    va_end(ap);
    return iret;
}

int
dxprintf(int fd, printf_domain_t __restrict domain, locale_t __restrict loc,
    const char * __restrict format, ...)
{
    int ret;
    va_list ap;

    va_start(ap, format);
    ret = _vdprintf(NULL, domain, fd, loc, format, ap);
    va_end(ap);
    return ret;
}

int
fxprintf(FILE * __restrict stream, printf_domain_t __restrict domain,
    locale_t __restrict loc, const char * __restrict format, ...)
{
    int ret;
    va_list ap;

    va_start(ap, format);
    ret = __xvprintf(NULL, domain, stream, loc, format, ap);
    va_end(ap);
    return ret;
}

int
sxprintf(char * __restrict str, size_t size, printf_domain_t __restrict domain,
    locale_t __restrict loc, const char * __restrict format, ...)
{
    int ret;
    va_list ap;

    va_start(ap, format);
    ret = _vsnprintf(NULL, domain, str, size, loc, format, ap);
    va_end(ap);
    return ret;
}

int
xprintf(printf_domain_t __restrict domain, locale_t __restrict loc,
    const char * __restrict format, ...)
{
    int ret;
    va_list ap;

    va_start(ap, format);
    ret = __xvprintf(NULL, domain, stdout, loc, format, ap);
    va_end(ap);
    return ret;
}

int
vasxprintf(char ** __restrict ret, printf_domain_t __restrict domain,
    locale_t __restrict loc, const char * __restrict format, va_list ap)
{
    return _vasprintf(NULL, domain, ret, loc, format, ap);
}

int
vdxprintf(int fd, printf_domain_t __restrict domain, locale_t __restrict loc,
    const char * __restrict format, va_list ap)
{
    return _vdprintf(NULL, domain, fd, loc, format, ap);
}

int
vfxprintf(FILE * __restrict stream, printf_domain_t __restrict domain,
    locale_t __restrict loc, const char * __restrict format, va_list ap)
{
    return __xvprintf(NULL, domain, stream, loc, format, ap);
}

int
vsxprintf(char * __restrict str, size_t size, printf_domain_t __restrict domain,
    locale_t __restrict loc, const char * __restrict format, va_list ap)
{
    return _vsnprintf(NULL, domain, str, size, loc, format, ap);
}

int
vxprintf(printf_domain_t __restrict domain, locale_t __restrict loc,
    const char * __restrict format, va_list ap)
{
    return __xvprintf(NULL, domain, stdout, loc, format, ap);
}
