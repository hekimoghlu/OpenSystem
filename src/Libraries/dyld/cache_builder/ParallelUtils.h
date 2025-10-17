/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 17, 2025.
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
#ifndef ParallelUtils_hpp
#define ParallelUtils_hpp

#include "Array.h"
#include "Error.h"

#include <dispatch/dispatch.h>
#include <span>
#include <vector>

namespace parallel
{

template<typename T>
static error::Error forEach(std::span<T> array, error::Error (^callback)(size_t index, T& element))
{
    const bool RunInSerial = false;

    dispatch_queue_t queue = RunInSerial ? dispatch_queue_create("serial", DISPATCH_QUEUE_SERIAL) : DISPATCH_APPLY_AUTO;

    BLOCK_ACCCESSIBLE_ARRAY(error::Error, errors, array.size());

    dispatch_apply(array.size(), queue, ^(size_t iteration) {
        T& element = array[iteration];
        errors[iteration] = callback(iteration, element);
    });

    if ( RunInSerial )
        dispatch_release(queue);

    // Return the first error we find
    for ( uint32_t i = 0; i != array.size(); ++i ) {
        error::Error& err = errors[i];
        if ( err.hasError() )
            return std::move(err);
    }

    return error::Error();
}

// Because "could not match 'span' against 'vector'", for some reason
template<typename T>
static error::Error forEach(std::vector<T>& array, error::Error (^callback)(size_t index, T& element))
{
    return forEach(std::span<T>(array), callback);
}

} // namespace parallel

#endif /* ParallelUtils_hpp */
