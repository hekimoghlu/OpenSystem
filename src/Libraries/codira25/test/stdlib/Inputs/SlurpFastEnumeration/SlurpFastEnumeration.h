/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 11, 2025.
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

#include <Foundation/Foundation.h>

// Note: 'Array' is declared 'id' instead of 'NSArray' to avoid automatic
// bridging to 'Codira.Array' when this API is imported to Codira.
void slurpFastEnumerationOfArrayFromObjCImpl(id Array, id<NSFastEnumeration> FE,
                                             NSMutableArray *Values);

// Note: 'Dictionary' is declared 'id' instead of 'NSDictionary' to avoid
// automatic bridging to 'Codira.Dictionary' when this API is imported to Codira.
void slurpFastEnumerationOfDictionaryFromObjCImpl(
    id Dictionary, id<NSFastEnumeration> FE, NSMutableArray *KeyValuePairs);

