/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 20, 2023.
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
// Copyright 2018 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// Observer:
//   Implements the Observer pattern for sending state change notifications
//   from Subject objects to dependent Observer objects.
//
//   See design document:
//   https://docs.google.com/document/d/15Edfotqg6_l1skTEL8ADQudF_oIdNa7i8Po43k6jMd4/

#include "libANGLE/Observer.h"

#include <algorithm>

#include "common/debug.h"

namespace angle
{
namespace
{}  // anonymous namespace

// Observer implementation.
ObserverInterface::~ObserverInterface() = default;

// Subject implementation.
Subject::Subject() {}

Subject::~Subject()
{
    resetObservers();
}

bool Subject::hasObservers() const
{
    return !mObservers.empty();
}

void Subject::onStateChange(SubjectMessage message) const
{
    if (mObservers.empty())
        return;

    for (const ObserverBindingBase *binding : mObservers)
    {
        binding->getObserver()->onSubjectStateChange(binding->getSubjectIndex(), message);
    }
}

void Subject::resetObservers()
{
    for (angle::ObserverBindingBase *binding : mObservers)
    {
        binding->onSubjectReset();
    }
    mObservers.clear();
}

// ObserverBinding implementation.
ObserverBinding::ObserverBinding() : ObserverBindingBase(nullptr, 0), mSubject(nullptr) {}

ObserverBinding::ObserverBinding(ObserverInterface *observer, SubjectIndex index)
    : ObserverBindingBase(observer, index), mSubject(nullptr)
{
    ASSERT(observer);
}

ObserverBinding::~ObserverBinding()
{
    reset();
}

ObserverBinding::ObserverBinding(const ObserverBinding &other)
    : ObserverBindingBase(other), mSubject(nullptr)
{
    bind(other.mSubject);
}

ObserverBinding &ObserverBinding::operator=(const ObserverBinding &other)
{
    reset();
    ObserverBindingBase::operator=(other);
    bind(other.mSubject);
    return *this;
}

void ObserverBinding::bind(Subject *subject)
{
    ASSERT(getObserver() || !subject);
    if (mSubject)
    {
        mSubject->removeObserver(this);
    }

    mSubject = subject;

    if (mSubject)
    {
        mSubject->addObserver(this);
    }
}

void ObserverBinding::onStateChange(SubjectMessage message) const
{
    getObserver()->onSubjectStateChange(getSubjectIndex(), message);
}

void ObserverBinding::onSubjectReset()
{
    mSubject = nullptr;
}
}  // namespace angle
