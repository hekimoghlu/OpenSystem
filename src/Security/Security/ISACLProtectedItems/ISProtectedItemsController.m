/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 20, 2023.
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
//  ISProtectedItemsController.m
//  ISACLProtectedItems
//
//  Copyright (c) 2014 Apple. All rights reserved.
//

#import "ISProtectedItemsController.h"
#import <spawn.h>

char * const pathToScrtiptFile = "/usr/local/bin/KeychainItemsAclTest.sh";

@implementation ISProtectedItemsController

- (NSArray *)specifiers
{
    if (!_specifiers) {
        _specifiers = [self loadSpecifiersFromPlistName:@"ISProtectedItems" target:self];
    }

    return _specifiers;
}

- (void)createBatchOfItems:(PSSpecifier *)specifier
{
    char * const argv[] = { pathToScrtiptFile,
                            "op=create",
                            NULL };

    posix_spawn(NULL, pathToScrtiptFile, NULL, NULL, argv, NULL);
}

- (void)deleteBatchOfItems:(PSSpecifier *)specifier
{
    char * const argv[] = { pathToScrtiptFile,
                            "op=delete",
                            NULL };

    posix_spawn(NULL, pathToScrtiptFile, NULL, NULL, argv, NULL);
}

@end
