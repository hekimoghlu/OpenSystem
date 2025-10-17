/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 20, 2022.
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
 * Copyright 2003 Sun Microsystems, Inc.  All rights reserved.
 * Use is subject to license terms.
 */

#ifndef _CTF_HEADERS_H
#define	_CTF_HEADERS_H

/*
 * Because the ON tools are executed on the system where they are built,
 * the tools need to include the headers installed on the build system,
 * rather than those in the ON source tree. However, some of the headers
 * required by the tools are part of the ON source tree, but not delivered
 * as part of Solaris.  These include the following:
 *
 * $(SRC)/lib/libctf/common/libctf.h
 * $(SRC)/uts/common/sys/ctf_api.h
 * $(SRC)/uts/common/sys/ctf.h
 *
 * These headers get installed in the proto area in the build environment
 * under $(ROOT)/usr/include and $(ROOT)/usr/include/sys. Though these
 * headers are not part of the release, in releases including and prior to
 * Solaris 9, they did get installed on the build system via bfu. Therefore,
 * we can not simply force the order of inclusion with -I/usr/include first
 * in Makefile.ctf because we might actually get downlevel versions of the
 * ctf headers. Depending on the order of the -I includes, we can also have
 * a problem with mismatched headers when building the ctf tools with some
 * headers getting pulled in from /usr/include and others from
 * $(SRC)/uts/common/sys.
 *
 * To address the problem, we have done two things:
 * 1) Created this header with a specific order of inclusion for the
 *    ctf headers.  Because the <libctf.h> header includes <sys/ctf_api.h>
 *    which in turn includes <sys/ctf.h> we need to include these in
 *    reverse order to guarantee that we get the correct versions of
 *    the headers.
 * 2) In $(SRC)/tools/ctf/Makefile.ctf, we order the -I includes such
 *    that we first search the directories where the ctf headers
 *    live, followed by /usr/include, followed by $(SRC)/uts/common.
 *    This last -I include is needed in order to prevent a build failure
 *    when <sys/ctf_api.h> is included via a nested #include rather than
 *    an explicit path #include.
 */

#if !defined(__APPLE__)
#include <uts/common/sys/ctf.h>
#include <uts/common/sys/ctf_api.h>
#include <lib/libctf/common/libctf.h>
#else
#include "darwin_shim.h"
#include "ctf.h"
#include "ctf_api.h"
#include "libctf.h"
#endif /* __APPLE__ */

#endif /* _CTF_HEADERS_H */
