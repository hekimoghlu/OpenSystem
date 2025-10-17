/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 4, 2025.
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

// Â© 2016 and later: Unicode, Inc. and others.
// License & terms of use: http://www.unicode.org/copyright.html
/*
 *******************************************************************************
 * Copyright (C) 2008, Google, International Business Machines Corporation and *
 * others. All Rights Reserved.                                                *
 *******************************************************************************
 */ 

#include "unicode/tmutamt.h"

#if !UCONFIG_NO_FORMATTING

U_NAMESPACE_BEGIN

UOBJECT_DEFINE_RTTI_IMPLEMENTATION(TimeUnitAmount)


TimeUnitAmount::TimeUnitAmount(const Formattable& number, 
                               TimeUnit::UTimeUnitFields timeUnitField,
                               UErrorCode& status)
:    Measure(number, TimeUnit::createInstance(timeUnitField, status), status) {
}


TimeUnitAmount::TimeUnitAmount(double amount, 
                               TimeUnit::UTimeUnitFields timeUnitField,
                               UErrorCode& status)
:   Measure(Formattable(amount), 
            TimeUnit::createInstance(timeUnitField, status),
            status) {
}


TimeUnitAmount::TimeUnitAmount(const TimeUnitAmount& other)
:   Measure(other)
{
}


TimeUnitAmount& 
TimeUnitAmount::operator=(const TimeUnitAmount& other) {
    Measure::operator=(other);
    return *this;
}


bool
TimeUnitAmount::operator==(const UObject& other) const {
    return Measure::operator==(other);
}

TimeUnitAmount* 
TimeUnitAmount::clone() const {
    return new TimeUnitAmount(*this);
}

    
TimeUnitAmount::~TimeUnitAmount() {
}



const TimeUnit&
TimeUnitAmount::getTimeUnit() const {
    return static_cast<const TimeUnit&>(getUnit());
}


TimeUnit::UTimeUnitFields
TimeUnitAmount::getTimeUnitField() const {
    return getTimeUnit().getTimeUnitField();
}
    

U_NAMESPACE_END

#endif
