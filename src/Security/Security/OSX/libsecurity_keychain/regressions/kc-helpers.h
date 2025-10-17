/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 18, 2022.
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
#ifndef kc_helpers_h
#define kc_helpers_h

#include <stdlib.h>
#include <unistd.h>

#include <Security/Security.h>
#include <Security/SecKeychainPriv.h>
#include "utilities/SecCFRelease.h"
#include "utilities/SecCFWrappers.h"

#include "kc-keychain-file-helpers.h"

extern char keychainFile[1000];
extern char keychainDbFile[1000];
extern char keychainTempFile[1000];
extern char keychainName[1000];
extern char testName[1000];

void startTest(const char* thisTestName);

void initializeKeychainTests(const char* thisTestName);

// Use this at the bottom of every test to make sure everything is gone
void deleteTestFiles(void);

void addToSearchList(SecKeychainRef keychain);

/* Checks to be sure there are N elements in this search, and returns the first
 * if it exists. */
SecKeychainItemRef checkNCopyFirst(char* testName, const CFDictionaryRef CF_CONSUMED query, uint32_t n);

void checkN(char* testName, const CFDictionaryRef CF_CONSUMED query, uint32_t n);
#define checkNTests 3

void readPasswordContentsWithResult(SecKeychainItemRef item, OSStatus expectedResult, CFStringRef expectedContents);
#define readPasswordContentsWithResultTests 3

void readPasswordContents(SecKeychainItemRef item, CFStringRef expectedContents);
#define readPasswordContentsTests readPasswordContentsWithResultTests

void changePasswordContents(SecKeychainItemRef item, CFStringRef newPassword);
#define changePasswordContentsTests 1

void deleteItem(SecKeychainItemRef item);
#define deleteItemTests 1

void deleteItems(CFArrayRef items);
#define deleteItemsTests 1

/* Checks in with securityd to see how many prompts were generated since the last call to this function, and tests against the number expected.
 Returns the number generated since the last call. */
uint32_t checkPrompts(uint32_t expectedSinceLastCall, char* explanation);
#define checkPromptsTests 2

#endif /* kc_helpers_h */
