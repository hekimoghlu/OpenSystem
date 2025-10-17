/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 13, 2023.
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
//
// mtl_resource_spi.h:
//    Used to set Metal resource ownership identity with SPI
//

#ifndef LIBANGLE_RENDERER_METAL_RESOURCE_SPI_H_
#define LIBANGLE_RENDERER_METAL_RESOURCE_SPI_H_

#import "common/apple/apple_platform.h"

#if ANGLE_USE_METAL_OWNERSHIP_IDENTITY

#    import <Metal/MTLResource_Private.h>
#    import <Metal/Metal.h>
#    import <mach/mach_types.h>

namespace rx
{
namespace mtl
{
inline void setOwnerWithIdentity(id<MTLResource> resource, task_id_token_t identityToken)
{
    if (identityToken != TASK_ID_TOKEN_NULL)
    {
        kern_return_t kr = [(id<MTLResourceSPI>)resource setOwnerWithIdentity:identityToken];
        if (ANGLE_UNLIKELY(kr != KERN_SUCCESS))
        {
            ERR() << "setOwnerWithIdentity failed with: %s (%x)" << mach_error_string(kr) << kr;
            ASSERT(false);
        }
    }
    return;
}
}  // namespace mtl
}  // namespace rx
#endif

#endif /* LIBANGLE_RENDERER_METAL_RESOURCE_SPI_H_ */
