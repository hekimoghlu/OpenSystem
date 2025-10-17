/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 22, 2023.
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
#ifndef __IMG4_OBJECT_H
#define __IMG4_OBJECT_H

#ifndef __IMG4_INDIRECT
#error "Please #include <img4/firmware.h> instead of this file directly"
#endif // __IMG4_INDIRECT

__BEGIN_DECLS
OS_ASSUME_NONNULL_BEGIN
OS_ASSUME_PTR_ABI_SINGLE_BEGIN

/*!
 * @typedef img4_object_spec_t
 * An opaque type which describes information about Image4 objects for use by
 * the runtime.
 */
IMG4_API_AVAILABLE_20200508
typedef struct _img4_object_spec img4_object_spec_t;

/*!
 * @const IMG4_FIRMWARE_SPEC
 * The object specification for an {@link img4_firmware_t} object.
 */
#if !XNU_KERNEL_PRIVATE
IMG4_API_AVAILABLE_20200508
OS_EXPORT
const img4_object_spec_t _img4_firmware_spec;
#define IMG4_FIRMWARE_SPEC (&_img4_firmware_spec)
#else
#define IMG4_FIRMWARE_SPEC (img4if->i4if_v7.firmware_spec)
#endif

/*!
 * @const IMG4_FIRMWARE_SIZE_RECOMMENDED
 * A constant describing the recommended stack allocation required for a
 * {@link img4_firmware_t} object.
 */
#define IMG4_FIRMWARE_SIZE_RECOMMENDED (1536u)

/*!
 * @const IMG4_CHIP_SPEC
 * The object specification for an {@link img4_chip_t} object.
 */
#if !XNU_KERNEL_PRIVATE
IMG4_API_AVAILABLE_20200508
OS_EXPORT
const img4_object_spec_t _img4_chip_spec;
#define IMG4_CHIP_SPEC (&_img4_chip_spec)
#else
#define IMG4_CHIP_SPEC (img4if->i4if_v7.chip_spec)
#endif

/*!
 * @const IMG4_CHIP_SIZE_RECOMMENDED
 * A constant describing the recommended stack allocation required for a
 * {@link img4_chip_t} object.
 */
#define IMG4_CHIP_SIZE_RECOMMENDED (960u)

/*!
 * @const IMG4_PMAP_DATA_SPEC
 * The object specification for an {@link img4_pmap_data_t} object.
 */
#if !XNU_KERNEL_PRIVATE
IMG4_API_AVAILABLE_20210521
OS_EXPORT
const img4_object_spec_t _img4_pmap_data_spec;
#define IMG4_PMAP_DATA_SPEC (&_img4_pmap_data_spec)
#else
#define IMG4_PMAP_DATA_SPEC (img4if->i4if_v13.pmap_data_spec)
#endif

/*!
 * @const IMG4_PMAP_DATA_SIZE_RECOMMENDED
 * A constant describing the recommended stack allocation required for a
 * {@link img4_pmap_data_t} object.
 */
#define IMG4_PMAP_DATA_SIZE_RECOMMENDED (5120u)

OS_ASSUME_PTR_ABI_SINGLE_END
OS_ASSUME_NONNULL_END
__END_DECLS

#endif // __IMG4_OBJECT_H
