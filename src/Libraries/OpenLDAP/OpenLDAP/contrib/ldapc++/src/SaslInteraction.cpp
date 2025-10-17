/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 12, 2025.
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

// $OpenLDAP$
/*
 * Copyright 2007-2011 The OpenLDAP Foundation, All Rights Reserved.
 * COPYING RESTRICTIONS APPLY, see COPYRIGHT file
 */

#include <SaslInteraction.h>
#include <iostream>
#include "debug.h"

SaslInteraction::SaslInteraction( sasl_interact_t *interact ) :
        m_interact(interact) {}

SaslInteraction::~SaslInteraction()
{
    DEBUG(LDAP_DEBUG_TRACE, "SaslInteraction::~SaslInteraction()" << std::endl);
}

unsigned long SaslInteraction::getId() const
{
    return m_interact->id;
}

const std::string SaslInteraction::getPrompt() const
{
    return std::string(m_interact->prompt);
}

const std::string SaslInteraction::getChallenge() const
{
    return std::string(m_interact->challenge);
}

const std::string SaslInteraction::getDefaultResult() const
{
    return std::string(m_interact->defresult);
}

void SaslInteraction::setResult(const std::string &res)
{
    m_result = res;
    m_interact->result = m_result.data();
    m_interact->len = m_result.size();
}
