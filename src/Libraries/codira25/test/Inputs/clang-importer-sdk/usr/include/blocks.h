/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 16, 2023.
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

@import Foundation;
@import Dispatch;

@interface NSString ()

- (void)enumerateLinesUsingBlock:
    (nonnull __attribute__((noescape)) void (^)(_Nonnull NSString *line))f;
// FIXME: The importer drops this.
//- (void)enumerateLinesUsingBlock:(void (^)(NSString *line, BOOL *b)) f;

@end

typedef void (^my_block_t)(void);

my_block_t blockWithoutNullability();
my_block_t _Nonnull blockWithNonnull();
my_block_t _Null_unspecified blockWithNullUnspecified();
my_block_t _Nullable blockWithNullable();

void accepts_block(my_block_t) __attribute__((nonnull));
void accepts_noescape_block(__attribute__((noescape)) my_block_t) __attribute__((nonnull));

// Please see related tests in PrintAsObjC/imported-block-typedefs.code.

