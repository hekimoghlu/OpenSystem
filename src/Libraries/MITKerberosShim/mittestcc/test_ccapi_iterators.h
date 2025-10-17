/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 26, 2022.
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

#ifndef _TEST_CCAPI_ITERATORS_H_
#define _TEST_CCAPI_ITERATORS_H_

#include "test_ccapi_globals.h"

int check_cc_ccache_iterator_next(void);
cc_int32 check_once_cc_ccache_iterator_next(cc_ccache_iterator_t iterator, cc_uint32 expected_count, cc_int32 expected_err, const char *description);

int check_cc_credentials_iterator_next(void);
cc_int32 check_once_cc_credentials_iterator_next(cc_credentials_iterator_t iterator, cc_uint32 expected_count, cc_int32 expected_err, const char *description);

#endif /* _TEST_CCAPI_ITERATORS_H_ */
