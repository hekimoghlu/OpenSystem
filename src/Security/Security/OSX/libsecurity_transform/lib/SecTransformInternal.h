/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 14, 2023.
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

#ifndef __SECTRANSFORM_INTERNAL__
#define __SECTRANSFORM_INTERNAL__

#include <Security/SecTransform.h>

#ifdef __cplusplus
extern "C" {
#endif

CFErrorRef SecTransformConnectTransformsInternal(SecGroupTransformRef groupRef, SecTransformRef sourceTransformRef, CFStringRef sourceAttributeName,
														 SecTransformRef destinationTransformRef, CFStringRef destinationAttributeName)
API_DEPRECATED("SecTransform is no longer supported", macos(10.7, 13.0)) API_UNAVAILABLE(ios, tvos, watchos, macCatalyst);

// note:  if destinationTransformRef is orphaned (i.e. left with nothing connecting to it and connecting to nothing, it will be removed
// from the group.
CFErrorRef SecTransformDisconnectTransforms(SecTransformRef destinationTransformRef, CFStringRef destinationAttributeName,
														 SecTransformRef sourceTransformRef, CFStringRef sourceAttributeName)
API_DEPRECATED("SecTransform is no longer supported", macos(10.7, 13.0)) API_UNAVAILABLE(ios, tvos, watchos, macCatalyst);

SecTransformRef SecGroupTransformFindLastTransform(SecGroupTransformRef groupTransform)
API_DEPRECATED("SecTransform is no longer supported", macos(10.7, 13.0)) API_UNAVAILABLE(ios, tvos, watchos, macCatalyst);

SecTransformRef SecGroupTransformFindMonitor(SecGroupTransformRef groupTransform)
API_DEPRECATED("SecTransform is no longer supported", macos(10.7, 13.0)) API_UNAVAILABLE(ios, tvos, watchos, macCatalyst);
bool SecGroupTransformHasMember(SecGroupTransformRef groupTransform, SecTransformRef transform)
API_DEPRECATED("SecTransform is no longer supported", macos(10.7, 13.0)) API_UNAVAILABLE(ios, tvos, watchos, macCatalyst);

CF_EXPORT
CFStringRef SecTransformDotForDebugging(SecTransformRef transformRef)
API_DEPRECATED("SecTransform is no longer supported", macos(10.7, 13.0)) API_UNAVAILABLE(ios, tvos, watchos, macCatalyst);
    
#ifdef __cplusplus
};
#endif

#endif
