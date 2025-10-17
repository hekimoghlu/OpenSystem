/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 6, 2024.
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

void jumpToLocation(double x, double y, double z);

void acceptDoublePointer(double* _Nonnull ptr) __attribute__((language_name("accept(_:)")));

void oldAcceptDoublePointer(double* _Nonnull ptr) __attribute__((availability(language, unavailable, replacement="acceptDoublePointer")));

void normallyUnchanged(void);
void normallyChangedOriginal(void) __attribute__((language_name("normallyChanged()")));

#ifdef __OBJC__

__attribute__((objc_root_class))
@interface A
@end

__attribute__((objc_root_class))
@interface TypeChanges
-(nonnull id)methodWithA:(nonnull id)a;
@end

__attribute__((objc_root_class))
@interface Base
-(nonnull instancetype)init;
@end

@interface B : A
@end

@interface C : B
@end

#endif // __OBJC__

#include <APINotesFrameworkTest/Classes.h>
#include <APINotesFrameworkTest/Enums.h>
#include <APINotesFrameworkTest/Globals.h>
#include <APINotesFrameworkTest/ImportAsMember.h>
#include <APINotesFrameworkTest/Properties.h>
#include <APINotesFrameworkTest/Protocols.h>
#include <APINotesFrameworkTest/Types.h>
#include <APINotesFrameworkTest/CodiraWrapper.h>
