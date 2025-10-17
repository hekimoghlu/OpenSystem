/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 8, 2025.
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
#include "config.h"
#include "CacheUpdate.h"

namespace JSC {

CacheUpdate::CacheUpdate(GlobalUpdate&& update)
    : m_update(WTFMove(update))
{
}

CacheUpdate::CacheUpdate(FunctionUpdate&& update)
    : m_update(WTFMove(update))
{
}

CacheUpdate::CacheUpdate(CacheUpdate&&) = default;

CacheUpdate& CacheUpdate::operator=(CacheUpdate&& other)
{
    this->~CacheUpdate();
    return *new (this) CacheUpdate(WTFMove(other));
}

bool CacheUpdate::isGlobal() const
{
    return std::holds_alternative<GlobalUpdate>(m_update);
}

const CacheUpdate::GlobalUpdate& CacheUpdate::asGlobal() const
{
    ASSERT(isGlobal());
    return std::get<GlobalUpdate>(m_update);
}

const CacheUpdate::FunctionUpdate& CacheUpdate::asFunction() const
{
    ASSERT(!isGlobal());
    return std::get<FunctionUpdate>(m_update);
}

} // namespace JSC
