/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 14, 2022.
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

#pragma clang assume_nonnull begin

int multiVersionedGlobal4;
int multiVersionedGlobal4Notes;
int multiVersionedGlobal4Header __attribute__((language_name("multiVersionedGlobal4Header_NEW")));
int multiVersionedGlobal4Both __attribute__((language_name("multiVersionedGlobal4Both_OLD")));

int multiVersionedGlobal34;
int multiVersionedGlobal34Notes;
int multiVersionedGlobal34Header __attribute__((language_name("multiVersionedGlobal34Header_NEW")));
int multiVersionedGlobal34Both __attribute__((language_name("multiVersionedGlobal34Both_OLD")));

int multiVersionedGlobal45;
int multiVersionedGlobal45Notes;
int multiVersionedGlobal45Header __attribute__((language_name("multiVersionedGlobal45Header_NEW")));
int multiVersionedGlobal45Both __attribute__((language_name("multiVersionedGlobal45Both_OLD")));

int multiVersionedGlobal345;
int multiVersionedGlobal345Notes;
int multiVersionedGlobal345Header __attribute__((language_name("multiVersionedGlobal345Header_NEW")));
int multiVersionedGlobal345Both __attribute__((language_name("multiVersionedGlobal345Both_OLD")));

int multiVersionedGlobal34_4_2;

#pragma clang assume_nonnull end
