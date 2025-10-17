/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 11, 2025.
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

// RUN: %empty-directory(%t)
// RUN: %target-clang %s -all_load %test-resource-dir/%target-sdk-name/liblanguageCompatibility50.a %test-resource-dir/%target-sdk-name/liblanguageCompatibility51.a -lobjc -o %t/main
// RUN: %target-codesign %t/main
// RUN: %target-run %t/main
// REQUIRES: objc_interop
// REQUIRES: executable_test

// The compatibility library needs to have no build-time dependencies on
// liblanguageCore so it can be linked into a program that doesn't link
// liblanguageCore, but will load it at runtime, such as xctest.
//
// Test this by linking it into a plain C program and making sure it builds.

int main(void) {}
