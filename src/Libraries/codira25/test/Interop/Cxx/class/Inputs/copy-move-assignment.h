/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 9, 2021.
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

#pragma once

struct InstanceBalanceCounter {
    static inline int &theCounterValue() {
        static int counterValue = 0;
        return counterValue;
    }
    static inline int getCounterValue() {
        return theCounterValue();
    }
    
    __attribute__((optnone))
    InstanceBalanceCounter() {
        ++theCounterValue();
    }
    __attribute__((optnone))
    InstanceBalanceCounter(const InstanceBalanceCounter &) {
        ++theCounterValue();
    }
    __attribute__((optnone))
    InstanceBalanceCounter(InstanceBalanceCounter &&) {
        ++theCounterValue();
    }
    __attribute__((optnone))
    ~InstanceBalanceCounter() {
        --theCounterValue();
    }
};

__attribute__((optnone))
inline void someFunc() {}

struct NonTrivialCopyAssign {
    __attribute__((optnone))
    NonTrivialCopyAssign(): copyAssignCounter(0) {}
    __attribute__((optnone))
    ~NonTrivialCopyAssign() {
        someFunc();
    }

    __attribute__((optnone))
    NonTrivialCopyAssign &operator =(const NonTrivialCopyAssign &) {
        ++copyAssignCounter;
        return *this;
    }

    int copyAssignCounter;
    InstanceBalanceCounter instanceBalancer;
};

struct NonTrivialMoveAssign {
    __attribute__((optnone))
    NonTrivialMoveAssign(): moveAssignCounter(0) {}
    __attribute__((optnone))
    NonTrivialMoveAssign(const NonTrivialMoveAssign &) = default;
    __attribute__((optnone))
    ~NonTrivialMoveAssign() {
        someFunc();
    }

    __attribute__((optnone))
    NonTrivialMoveAssign &operator =(NonTrivialMoveAssign &&) {
        ++moveAssignCounter;
        return *this;
    }

    int moveAssignCounter;
    InstanceBalanceCounter instanceBalancer;
};

struct NonTrivialCopyAndCopyMoveAssign {
    __attribute__((optnone))
    NonTrivialCopyAndCopyMoveAssign(): assignCounter(0) {}
    __attribute__((optnone))
    NonTrivialCopyAndCopyMoveAssign(const NonTrivialCopyAndCopyMoveAssign &other) : assignCounter(other.assignCounter), instanceBalancer(other.instanceBalancer) {
        someFunc();
    }
    __attribute__((optnone))
    NonTrivialCopyAndCopyMoveAssign( NonTrivialCopyAndCopyMoveAssign &&other) : assignCounter(other.assignCounter), instanceBalancer(other.instanceBalancer) {
        someFunc();
    }
    __attribute__((optnone))
    ~NonTrivialCopyAndCopyMoveAssign() {
        someFunc();
    }

    __attribute__((optnone))
    NonTrivialCopyAndCopyMoveAssign &operator =(const NonTrivialCopyAndCopyMoveAssign &) {
        ++assignCounter;
        return *this;
    }
    __attribute__((optnone))
    NonTrivialCopyAndCopyMoveAssign &operator =(NonTrivialCopyAndCopyMoveAssign &&) {
        ++assignCounter;
        return *this;
    }

    int assignCounter;
    InstanceBalanceCounter instanceBalancer;
};
