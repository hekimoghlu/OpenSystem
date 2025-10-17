/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 20, 2022.
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
#ifndef _SECTION_KEYWORDS_H
#define _SECTION_KEYWORDS_H

#define __PLACE_IN_SECTION(__segment__section) \
	__attribute__((used, section(__segment__section)))

#define __SEGMENT_START_SYM(seg)       asm("segment$start$" seg)
#define __SEGMENT_END_SYM(seg)         asm("segment$end$" seg)

#define __SECTION_START_SYM(seg, sect) asm("section$start$" seg "$" sect)
#define __SECTION_END_SYM(seg, sect)   asm("section$end$" seg "$" sect)

#if defined(__arm64__) || defined (__x86_64__)

#define SECURITY_SEGMENT_NAME           "__DATA"
#define SECURITY_SECTION_NAME           "__const"
#define SECURITY_SEGMENT_SECTION_NAME   "__DATA,__const"

#define __security_const_early const
#define __security_const_late __attribute__((section(SECURITY_SEGMENT_SECTION_NAME)))
#define __security_read_write

#if HIBERNATION
#define MARK_AS_HIBERNATE_TEXT __attribute__((section("__HIB, __text, regular, pure_instructions")))
#define MARK_AS_HIBERNATE_DATA __attribute__((section("__HIB, __data")))
#define MARK_AS_HIBERNATE_DATA_CONST_LATE __attribute__((section("__HIB, __const")))
#endif /* HIBERNATION */
#endif /* __arm64__ || __x86_64__ */

#ifndef __security_const_early
#define __security_const_early const
#endif
#ifndef __security_const_late
#define __security_const_late
#endif
#ifndef __security_read_write
#define __security_read_write
#endif
#ifndef MARK_AS_HIBERNATE_TEXT
#define MARK_AS_HIBERNATE_TEXT
#endif
#ifndef MARK_AS_HIBERNATE_DATA
#define MARK_AS_HIBERNATE_DATA
#endif
#ifndef MARK_AS_HIBERNATE_DATA_CONST_LATE
#define MARK_AS_HIBERNATE_DATA_CONST_LATE
#endif

#define SECURITY_READ_ONLY_EARLY(_t) _t __security_const_early __attribute__((used))
#define SECURITY_READ_ONLY_LATE(_t)  _t __security_const_late  __attribute__((used))
#define SECURITY_READ_WRITE(_t)      _t __security_read_write  __attribute__((used))

#if CONFIG_SPTM
/*
 * Place a function in a special segment, __TEXT_BOOT_EXEC. Code placed
 * in this segment will be allowed by the SPTM to execute during the fixups
 * phase; the rest of the code will be mapped as RW, so that it can be overwritten.
 * Code that is required to execute in order to apply fixups MUST be contained
 * in this special segment.
 */
#define MARK_AS_FIXUP_TEXT __attribute__((used, section("__TEXT_BOOT_EXEC,__bootcode,regular,pure_instructions")))
#else
#define MARK_AS_FIXUP_TEXT
#endif

#endif /* _SECTION_KEYWORDS_H_ */
