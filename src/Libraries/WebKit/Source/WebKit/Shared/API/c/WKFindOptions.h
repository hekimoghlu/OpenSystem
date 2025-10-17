/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 16, 2025.
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
#ifndef WKFindOptions_h
#define WKFindOptions_h

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    kWKFindOptionsCaseInsensitive = 1 << 0,
    kWKFindOptionsAtWordStarts = 1 << 1,
    kWKFindOptionsTreatMedialCapitalAsWordStart = 1 << 2,
    kWKFindOptionsBackwards = 1 << 3,
    kWKFindOptionsWrapAround = 1 << 4,
    kWKFindOptionsShowOverlay = 1 << 5,
    kWKFindOptionsShowFindIndicator = 1 << 6,
    kWKFindOptionsShowHighlight = 1 << 7
};
typedef uint32_t WKFindOptions;

enum { kWKFindResultNoMatchAfterUserSelection = -1 };

#ifdef __cplusplus
}
#endif

#endif // WKFindOptions_h
