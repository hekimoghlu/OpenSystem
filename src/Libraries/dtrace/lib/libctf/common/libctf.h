/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 22, 2022.
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
/*
 * Copyright 2001-2003 Sun Microsystems, Inc.  All rights reserved.
 * Use is subject to license terms.
 */

/*
 * This header file defines the interfaces available from the CTF debugger
 * library, libctf.  This library provides functions that a debugger can
 * use to operate on data in the Compact ANSI-C Type Format (CTF).  This
 * is NOT a public interface, although it may eventually become one in
 * the fullness of time after we gain more experience with the interfaces.
 *
 * In the meantime, be aware that any program linked with libctf in this
 * release of Solaris is almost guaranteed to break in the next release.
 *
 * In short, do not user this header file or libctf for any purpose.
 */

#ifndef	_LIBCTF_H
#define	_LIBCTF_H

#pragma ident	"@(#)libctf.h	1.6	05/06/08 SMI"

#include <ctf_api.h>

#ifdef	__cplusplus
extern "C" {
#endif

/*
 * This flag can be used to enable debug messages.
 */
extern int _libctf_debug;

#ifdef	__cplusplus
}
#endif

#endif	/* _LIBCTF_H */
