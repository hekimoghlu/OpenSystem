/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 26, 2025.
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

// $OpenLDAP$
/*
 * Copyright 2000-2011 The OpenLDAP Foundation, All Rights Reserved.
 * COPYING RESTRICTIONS APPLY, see COPYRIGHT file
 */

#ifndef DEBUG_H
#define DEBUG_H
#include <iostream>
#include "config.h"
#define LDAP_DEBUG_NONE         0x0000
#define LDAP_DEBUG_TRACE        0x0001
#define LDAP_DEBUG_CONSTRUCT    0x0002
#define LDAP_DEBUG_DESTROY      0x0004
#define LDAP_DEBUG_PARAMETER    0x0008
#define LDAP_DEBUG_ANY          0xffff 

#define DEBUGLEVEL LDAP_DEBUG_ANY

#define PRINT_FILE	\
	std::cerr << "file: " __FILE__  << " line: " << __LINE__ 

#ifdef WITH_DEBUG
#define DEBUG(level, arg)       \
    if((level) & DEBUGLEVEL){     \
        std::cerr  << arg ;          \
    } 
#else
#undef DEBUG
#define DEBUG(level,arg)
#endif //WITH_DEBUG

#endif // DEBUG_H
