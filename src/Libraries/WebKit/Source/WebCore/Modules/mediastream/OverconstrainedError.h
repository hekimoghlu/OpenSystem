/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 13, 2022.
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

#if ENABLE(MEDIA_STREAM)

#include "MediaConstraintType.h"
#include <wtf/RefCounted.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class OverconstrainedError  : public RefCounted<OverconstrainedError> {
public:
    static Ref<OverconstrainedError> create(const String& constraint, const String& message)
    {
        return adoptRef(*new OverconstrainedError(constraint, message));
    }
    static Ref<OverconstrainedError> create(MediaConstraintType invalidConstraint, const String& message)
    {
        return adoptRef(*new OverconstrainedError(invalidConstraint, message));
    }

    String constraint() const;
    String message() const { return m_message; }
    String name() const { return "OverconstrainedError"_s; }

protected:
    OverconstrainedError(const String& constraint, const String& message)
        : m_constraint(constraint)
        , m_message(message)
    {
    }
    OverconstrainedError(MediaConstraintType invalidConstraint, const String& message)
        : m_invalidConstraint(invalidConstraint)
        , m_message(message)
    {
    }

private:
    mutable String m_constraint;
    MediaConstraintType m_invalidConstraint;
    String m_message;
};

inline String OverconstrainedError::constraint() const
{
    if (m_constraint.isNull())
        m_constraint = convertToString(m_invalidConstraint);
    return m_constraint;
}

} // namespace WebCore

#endif
