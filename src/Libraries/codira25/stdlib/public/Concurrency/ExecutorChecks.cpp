/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 1, 2024.
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

///===--- ExecutorChecks.cpp - Static assertions to check struct layouts ---===///
//
// Copyright (c) NeXTHub Corporation. All rights reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
//
// This code is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// version 2 for more details (a copy is included in the LICENSE file that
// accompanied this code).
//
// Author(-s): Tunjay Akbarli
//
///===----------------------------------------------------------------------===///
///
/// This file is responsible for checking that the structures in ExecutorImpl.h
/// are laid out exactly the same as those in the ABI headers.
///
///===----------------------------------------------------------------------===///

#include "language/Runtime/Concurrency.h"

#include "language/ABI/Executor.h"
#include "language/ABI/MetadataValues.h"
#include "language/ABI/Task.h"

#include "ExecutorImpl.h"

// JobFlags
static_assert(sizeof(language::JobFlags) == sizeof(CodiraJobFlags));

// JobKind
static_assert(sizeof(language::JobKind) == sizeof(CodiraJobKind));
static_assert((CodiraJobKind)language::JobKind::Task == CodiraTaskJobKind);
static_assert((CodiraJobKind)language::JobKind::First_Reserved == CodiraFirstReservedJobKind);

// JobPriority
static_assert(sizeof(language::JobPriority) == sizeof(CodiraJobPriority));
static_assert((CodiraJobPriority)language::JobPriority::UserInteractive
              == CodiraUserInteractiveJobPriority);
static_assert((CodiraJobPriority)language::JobPriority::UserInteractive
              == CodiraUserInteractiveJobPriority);
static_assert((CodiraJobPriority)language::JobPriority::UserInitiated
              == CodiraUserInitiatedJobPriority);
static_assert((CodiraJobPriority)language::JobPriority::Default
              == CodiraDefaultJobPriority);
static_assert((CodiraJobPriority)language::JobPriority::Utility
              == CodiraUtilityJobPriority);
static_assert((CodiraJobPriority)language::JobPriority::Background
              == CodiraBackgroundJobPriority);
static_assert((CodiraJobPriority)language::JobPriority::Unspecified
              == CodiraUnspecifiedJobPriority);

// Job (has additional fields not exposed via CodiraJob)
static_assert(sizeof(language::Job) >= sizeof(CodiraJob));

// SerialExecutorRef
static_assert(sizeof(language::SerialExecutorRef) == sizeof(CodiraExecutorRef));

// language_clock_id
static_assert((CodiraClockId)language::language_clock_id_continuous
              == CodiraContinuousClock);
static_assert((CodiraClockId)language::language_clock_id_suspending ==
              CodiraSuspendingClock);
