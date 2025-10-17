/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 17, 2021.
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

import ICU_Private

func setupICUMallocZone() {
    let zone = malloc_create_zone(0, 0)
    malloc_set_zone_name(zone, "ICU")

    var status = U_ZERO_ERROR
    u_setMemoryFunctions(zone, icuAlloc, icuRealloc, icuFree, &status)
}

// typedef void *U_CALLCONV UMemAllocFn(const void *context, size_t size);
private func icuAlloc(zone: UnsafeRawPointer?, size: Int) -> UnsafeMutableRawPointer? {
    guard let zone = UnsafeMutableRawPointer(mutating: zone) else {
        fatalError("missing malloc zone")
    }
    return malloc_zone_malloc(zone.bindMemory(to: malloc_zone_t.self, capacity: 1), size)
}

// typedef void *U_CALLCONV UMemReallocFn(const void *context, void *mem, size_t size);
private func icuRealloc(zone: UnsafeRawPointer?, ptr: UnsafeMutableRawPointer?, size: Int) -> UnsafeMutableRawPointer? {
    guard let zone = UnsafeMutableRawPointer(mutating: zone) else {
        fatalError("missing malloc zone")
    }
    return malloc_zone_realloc(zone.bindMemory(to: malloc_zone_t.self, capacity: 1), ptr, size)
}

// typedef void  U_CALLCONV UMemFreeFn (const void *context, void *mem);
private func icuFree(zone: UnsafeRawPointer?, ptr: UnsafeMutableRawPointer?) {
    guard let zone = UnsafeMutableRawPointer(mutating: zone) else {
        fatalError("missing malloc zone")
    }
    malloc_zone_free(zone.bindMemory(to: malloc_zone_t.self, capacity: 1), ptr)
}
