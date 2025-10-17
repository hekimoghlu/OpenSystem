/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 20, 2025.
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
#ifndef ND_GETSERVENT_H
#define ND_GETSERVENT_H

#ifdef _NETDB_H_
/* Just in case... */
#error netdb.h and getservent.h are incompatible
#else
#define _NETDB_H_
#endif

#ifdef _WIN32
#define __PATH_SYSROOT "SYSTEMROOT"
#define __PATH_ETC_INET "\\System32\\drivers\\etc\\"
#define __PATH_SERVICES "services"
#else
/*
* The idea here is to be able to replace "PREFIX" in __PATH_SYSROOT with a variable
* that could, for example, point to an alternative install location.
*/
#define __PATH_SYSROOT "PREFIX"
#define __PATH_ETC_INET "/etc/"
#define __PATH_SERVICES __PATH_ETC_INET"services"
#endif

#define MAXALIASES 35

void endservent (void);
struct servent *getservent(void);
void setservent (int f);

#endif /* ! ND_GETSERVENT_H */
