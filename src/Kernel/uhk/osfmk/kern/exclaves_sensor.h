/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 8, 2022.
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
#if CONFIG_EXCLAVES

#pragma once

#include <mach/exclaves.h>
#include <mach/kern_return.h>

#include <stdint.h>

__BEGIN_DECLS

/*!
 * @function exclaves_sensor_copy
 *
 * @abstract
 * Allow a copy from an aribtrated audio memory segment
 *
 * @param buffer
 * Identifies which arbitrated memory buffer to operate on
 *
 * @param size1
 * The length in bytes of the data to be copied
 *
 * @param offset1
 * Offset in bytes of the data to be copied
 *
 * @param size2
 * The length in bytes of the data to be copied
 *
 * @param offset2
 * Offset in bytes of the data to be copied
 *
 * @param sensor_status
 * Out parameter filled with the sensor status.
 *
 * @result
 * KERN_SUCCESS or mach system call error code.
 */
kern_return_t
exclaves_sensor_copy(uint32_t buffer, uint64_t size1,
    uint64_t offset1, uint64_t size2, uint64_t offset2,
    exclaves_sensor_status_t *sensor_status);

__END_DECLS

#endif /* CONFIG_EXCLAVES */
