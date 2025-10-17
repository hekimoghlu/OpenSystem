/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 20, 2023.
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
#include <errno.h>
#include <local.h>
#include <xprintf_private.h>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpointer-bool-conversion"

int
asxprintf_exec(char ** __restrict ret,
    printf_comp_t __restrict pc, ...)
{
    int iret;
    va_list ap;

    if (!pc) {
	errno = EINVAL;
	return -1;
    }

    va_start(ap, pc);
    iret = _vasprintf(pc, NULL, ret, NULL, NULL, ap);
    va_end(ap);
    return iret;
}

int
dxprintf_exec(int fd, printf_comp_t __restrict pc, ...)
{
    int ret;
    va_list ap;

    if (!pc) {
	errno = EINVAL;
	return -1;
    }

    va_start(ap, pc);
    ret = _vdprintf(pc, NULL, fd, NULL, NULL, ap);
    va_end(ap);
    return ret;
}

int
fxprintf_exec(FILE * __restrict stream,
    printf_comp_t __restrict pc, ...)
{
    int ret;
    va_list ap;

    if (!pc) {
	errno = EINVAL;
	return -1;
    }

    va_start(ap, pc);
    ret = __xvprintf(pc, NULL, stream, NULL, NULL, ap);
    va_end(ap);
    return ret;
}

int
sxprintf_exec(char * __restrict str, size_t size,
    printf_comp_t __restrict pc, ...)
{
    int ret;
    va_list ap;

    if (!pc) {
	errno = EINVAL;
	return -1;
    }

    va_start(ap, pc);
    ret = _vsnprintf(pc, NULL, str, size, NULL, NULL, ap);
    va_end(ap);
    return ret;
}

int
xprintf_exec(printf_comp_t __restrict pc, ...)
{
    int ret;
    va_list ap;

    if (!pc) {
	errno = EINVAL;
	return -1;
    }

    va_start(ap, pc);
    ret = __xvprintf(pc, NULL, stdout, NULL, NULL, ap);
    va_end(ap);
    return ret;
}

int
vasxprintf_exec(char ** __restrict ret,
    printf_comp_t __restrict pc, va_list ap)
{
    if (!pc) {
	errno = EINVAL;
	return -1;
    }

    return _vasprintf(pc, NULL, ret, NULL, NULL, ap);
}

int
vdxprintf_exec(int fd, printf_comp_t __restrict pc,
    va_list ap) 
{
    if (!pc) {
	errno = EINVAL;
	return -1;
    }

    return _vdprintf(pc, NULL, fd, NULL, NULL, ap);
}

int
vfxprintf_exec(FILE * __restrict stream,
    printf_comp_t __restrict pc, va_list ap)
{
    if (!pc) {
	errno = EINVAL;
	return -1;
    }

    return __xvprintf(pc, NULL, stream, NULL, NULL, ap);
}

int
vsxprintf_exec(char * __restrict str, size_t size,
    printf_comp_t __restrict pc, va_list ap)
{
    if (!pc) {
	errno = EINVAL;
	return -1;
    }

    return _vsnprintf(pc, NULL, str, size, NULL, NULL, ap);
}

int
vxprintf_exec(printf_comp_t __restrict pc, va_list ap)
{
    if (!pc) {
	errno = EINVAL;
	return -1;
    }

    return __xvprintf(pc, NULL, stdout, NULL, NULL, ap);
}
#pragma clang diagnostic pop
