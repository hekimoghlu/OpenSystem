/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 9, 2025.
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
#include "InjectedBundleScriptWorld.h"

#include <WebCore/DOMWrapperWorld.h>
#include <WebCore/ScriptController.h>
#include <wtf/CheckedPtr.h>
#include <wtf/HashMap.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/text/MakeString.h>
#include <wtf/text/WTFString.h>

namespace WebKit {
using namespace WebCore;

using WorldMap = HashMap<SingleThreadWeakRef<DOMWrapperWorld>, WeakRef<InjectedBundleScriptWorld>>;

static WorldMap& allWorlds()
{
    static NeverDestroyed<WorldMap> map;
    return map;
}

static String uniqueWorldName()
{
    static uint64_t uniqueWorldNameNumber = 0;
    return makeString("UniqueWorld_"_s, uniqueWorldNameNumber++);
}

Ref<InjectedBundleScriptWorld> InjectedBundleScriptWorld::create(Type type)
{
    return InjectedBundleScriptWorld::create(uniqueWorldName(), type);
}

Ref<InjectedBundleScriptWorld> InjectedBundleScriptWorld::create(const String& name, Type type)
{
    return adoptRef(*new InjectedBundleScriptWorld(ScriptController::createWorld(name, type == Type::User ? ScriptController::WorldType::User : ScriptController::WorldType::Internal), name));
}

Ref<InjectedBundleScriptWorld> InjectedBundleScriptWorld::getOrCreate(DOMWrapperWorld& world)
{
    if (&world == &mainThreadNormalWorld())
        return normalWorld();

    if (auto existingWorld = allWorlds().get(world))
        return *existingWorld;

    return adoptRef(*new InjectedBundleScriptWorld(world, uniqueWorldName()));
}

InjectedBundleScriptWorld* InjectedBundleScriptWorld::find(const String& name)
{
    for (auto& world : allWorlds().values()) {
        if (world->name() == name)
            return world.ptr();
    }
    return nullptr;
}

InjectedBundleScriptWorld& InjectedBundleScriptWorld::normalWorld()
{
    static InjectedBundleScriptWorld& world = adoptRef(*new InjectedBundleScriptWorld(mainThreadNormalWorld(), String())).leakRef();
    return world;
}

InjectedBundleScriptWorld::InjectedBundleScriptWorld(DOMWrapperWorld& world, const String& name)
    : m_world(world)
    , m_name(name)
{
    ASSERT(!allWorlds().contains(world));
    allWorlds().add(world, *this);
}

InjectedBundleScriptWorld::~InjectedBundleScriptWorld()
{
    ASSERT(allWorlds().contains(m_world.get()));
    allWorlds().remove(m_world.get());
}

const DOMWrapperWorld& InjectedBundleScriptWorld::coreWorld() const
{
    return m_world;
}

DOMWrapperWorld& InjectedBundleScriptWorld::coreWorld()
{
    return m_world;
}
    
void InjectedBundleScriptWorld::clearWrappers()
{
    m_world->clearWrappers();
}

void InjectedBundleScriptWorld::setAllowAutofill()
{
    m_world->setAllowAutofill();
}

void InjectedBundleScriptWorld::setAllowElementUserInfo()
{
    m_world->setAllowElementUserInfo();
}

void InjectedBundleScriptWorld::makeAllShadowRootsOpen()
{
    m_world->setShadowRootIsAlwaysOpen();
}

void InjectedBundleScriptWorld::disableOverrideBuiltinsBehavior()
{
    m_world->disableLegacyOverrideBuiltInsBehavior();
}

} // namespace WebKit
