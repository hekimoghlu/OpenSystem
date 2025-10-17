/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 8, 2022.
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

#ifndef __COLLECTIONS__
#define __COLLECTIONS__

#ifndef __MACTYPES__
#include <MacTypes.h>
#endif

enum {
  kCollectionNoAttributes       = 0x00000000,
  kCollectionAllAttributes      = (int)0xFFFFFFFF,
  kCollectionUserAttributes     = 0x0000FFFF,
  kCollectionDefaultAttributes  = 0x40000000
};

/* abstract data type for a collection */
typedef struct OpaqueCollection*        Collection;
/* collection member 4 byte tag */
typedef FourCharCode                    CollectionTag;

#endif
