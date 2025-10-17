/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 28, 2023.
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

#ifndef TZCONFIG_H_INCLUDED
#define TZCONFIG_H_INCLUDED

#define TM_GMTOFF	tm_gmtoff
#define TM_ZONE		tm_zone

#define HAVE_GETTEXT	false
#define HAVE_SYS_STAT_H	true
#define HAVE_UNISTD_H	true
#define HAVE_STDINT_H	true

#define PCTS		1
#define NETBSD_INSPIRED	0
#define STD_INSPIRED	1
#define HAVE_TZNAME	2
#define USG_COMPAT	0
#define ALTZONE		0

#endif /* !TZCONFIG_H_INCLUDED */
