/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 15, 2025.
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
#include "WebKitScriptWorld.h"

#include "WebKitScriptWorldPrivate.h"
#include <wtf/HashMap.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/glib/WTFGType.h>

using namespace WebKit;
using namespace WebCore;

enum {
    WINDOW_OBJECT_CLEARED,

    LAST_SIGNAL
};

typedef HashMap<InjectedBundleScriptWorld*, WebKitScriptWorld*> ScriptWorldMap;

static ScriptWorldMap& scriptWorlds()
{
    static NeverDestroyed<ScriptWorldMap> map;
    return map;
}

struct _WebKitScriptWorldPrivate {
    ~_WebKitScriptWorldPrivate()
    {
        ASSERT(scriptWorlds().contains(scriptWorld.get()));
        scriptWorlds().remove(scriptWorld.get());
    }

    RefPtr<InjectedBundleScriptWorld> scriptWorld;
    CString name;
};

static std::array<unsigned, LAST_SIGNAL> signals;

WEBKIT_DEFINE_FINAL_TYPE(WebKitScriptWorld, webkit_script_world, G_TYPE_OBJECT, GObject)

static void webkit_script_world_class_init(WebKitScriptWorldClass* klass)
{
    /**
     * WebKitScriptWorld::window-object-cleared:
     * @world: the #WebKitScriptWorld on which the signal is emitted
     * @page: a #WebKitWebPage
     * @frame: the #WebKitFrame  to which @world belongs
     *
     * Emitted when the JavaScript window object in a #WebKitScriptWorld has been
     * cleared. This is the preferred place to set custom properties on the window
     * object using the JavaScriptCore API. You can get the window object of @frame
     * from the JavaScript execution context of @world that is returned by
     * webkit_frame_get_js_context_for_script_world().
     *
     * Since: 2.2
     */
    signals[WINDOW_OBJECT_CLEARED] = g_signal_new(
        "window-object-cleared",
        G_TYPE_FROM_CLASS(klass),
        G_SIGNAL_RUN_LAST,
        0, nullptr, nullptr,
        g_cclosure_marshal_generic,
        G_TYPE_NONE, 2,
        WEBKIT_TYPE_WEB_PAGE,
        WEBKIT_TYPE_FRAME);
}

WebKitScriptWorld* webkitScriptWorldGet(InjectedBundleScriptWorld* scriptWorld)
{
    return scriptWorlds().get(scriptWorld);
}

InjectedBundleScriptWorld* webkitScriptWorldGetInjectedBundleScriptWorld(WebKitScriptWorld* world)
{
    return world->priv->scriptWorld.get();
}

void webkitScriptWorldWindowObjectCleared(WebKitScriptWorld* world, WebKitWebPage* page, WebKitFrame* frame)
{
    g_signal_emit(world, signals[WINDOW_OBJECT_CLEARED], 0, page, frame);
}

static WebKitScriptWorld* webkitScriptWorldCreate(Ref<InjectedBundleScriptWorld>&& scriptWorld)
{
    WebKitScriptWorld* world = WEBKIT_SCRIPT_WORLD(g_object_new(WEBKIT_TYPE_SCRIPT_WORLD, nullptr));
    world->priv->scriptWorld = WTFMove(scriptWorld);
    world->priv->name = world->priv->scriptWorld->name().utf8();

    ASSERT(!scriptWorlds().contains(world->priv->scriptWorld.get()));
    scriptWorlds().add(world->priv->scriptWorld.get(), world);

    return world;
}

static gpointer createDefaultScriptWorld(gpointer)
{
    return webkitScriptWorldCreate(InjectedBundleScriptWorld::normalWorld());
}

/**
 * webkit_script_world_get_default:
 *
 * Get the default #WebKitScriptWorld. This is the normal script world
 * where all scripts are executed by default.
 * You can get the JavaScript execution context of a #WebKitScriptWorld
 * for a given #WebKitFrame with webkit_frame_get_javascript_context_for_script_world().
 *
 * Returns: (transfer none): the default #WebKitScriptWorld
 *
 * Since: 2.2
 */
WebKitScriptWorld* webkit_script_world_get_default(void)
{
    static GOnce onceInit = G_ONCE_INIT;
    return WEBKIT_SCRIPT_WORLD(g_once(&onceInit, createDefaultScriptWorld, 0));
}

/**
 * webkit_script_world_new:
 *
 * Creates a new isolated #WebKitScriptWorld. Scripts executed in
 * isolated worlds have access to the DOM but not to other variable
 * or functions created by the page.
 * The #WebKitScriptWorld is created with a generated unique name. Use
 * webkit_script_world_new_with_name() if you want to create it with a
 * custom name.
 * You can get the JavaScript execution context of a #WebKitScriptWorld
 * for a given #WebKitFrame with webkit_frame_get_javascript_context_for_script_world().
 *
 * Returns: (transfer full): a new isolated #WebKitScriptWorld
 *
 * Since: 2.2
 */
WebKitScriptWorld* webkit_script_world_new(void)
{
    return webkitScriptWorldCreate(InjectedBundleScriptWorld::create(InjectedBundleScriptWorld::Type::User));
}

/**
 * webkit_script_world_new_with_name:
 * @name: a name for the script world
 *
 * Creates a new isolated #WebKitScriptWorld with a name. Scripts executed in
 * isolated worlds have access to the DOM but not to other variable
 * or functions created by the page.
 * You can get the JavaScript execution context of a #WebKitScriptWorld
 * for a given #WebKitFrame with webkit_frame_get_javascript_context_for_script_world().
 *
 * Returns: (transfer full): a new isolated #WebKitScriptWorld
 *
 * Since: 2.22
 */
WebKitScriptWorld* webkit_script_world_new_with_name(const char* name)
{
    g_return_val_if_fail(name, nullptr);

    return webkitScriptWorldCreate(InjectedBundleScriptWorld::create(String::fromUTF8(name), InjectedBundleScriptWorld::Type::User));
}

/**
 * webkit_script_world_get_name:
 * @world: a #WebKitScriptWorld
 *
 * Get the name of a #WebKitScriptWorld.
 *
 * Returns: the name of @world
 *
 * Since: 2.22
 */
const char* webkit_script_world_get_name(WebKitScriptWorld* world)
{
    g_return_val_if_fail(WEBKIT_IS_SCRIPT_WORLD(world), nullptr);

    return world->priv->name.data();
}
