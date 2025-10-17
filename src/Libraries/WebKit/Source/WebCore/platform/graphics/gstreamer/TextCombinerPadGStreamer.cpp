/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 15, 2023.
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
#include "TextCombinerPadGStreamer.h"

#if ENABLE(VIDEO) && USE(GSTREAMER)

#include "GStreamerCommon.h"
#include "TextCombinerGStreamer.h"
#include <wtf/glib/WTFGType.h>

struct _WebKitTextCombinerPadPrivate {
    GRefPtr<GstTagList> tags;
    GRefPtr<GstPad> innerCombinerPad;
    bool shouldProcessStickyEvents { true };
};

enum {
    PROP_PAD_0,
    PROP_PAD_TAGS,
    PROP_INNER_COMBINER_PAD,
    N_PROPERTIES,
};

static std::array<GParamSpec*, N_PROPERTIES> sObjProperties;

WEBKIT_DEFINE_TYPE(WebKitTextCombinerPad, webkit_text_combiner_pad, GST_TYPE_GHOST_PAD);

static gboolean webkitTextCombinerPadEvent(GstPad* pad, GstObject* parent, GstEvent* event)
{
    switch (GST_EVENT_TYPE(event)) {
    case GST_EVENT_TAG: {
        auto* combinerPad = WEBKIT_TEXT_COMBINER_PAD(pad);
        GstTagList* tags;
        gst_event_parse_tag(event, &tags);
        ASSERT(tags);

        {
            auto locker = GstObjectLocker(pad);
            if (!combinerPad->priv->tags)
                combinerPad->priv->tags = adoptGRef(gst_tag_list_copy(tags));
            else
                gst_tag_list_insert(combinerPad->priv->tags.get(), tags, GST_TAG_MERGE_REPLACE);
        }

        g_object_notify_by_pspec(G_OBJECT(pad), sObjProperties[PROP_PAD_TAGS]);
        break;
    }
    default:
        break;
    }
    return gst_pad_event_default(pad, parent, event);
}

static GstFlowReturn webkitTextCombinerPadChain(GstPad* pad, GstObject* parent, GstBuffer* buffer)
{
    auto* combinerPad = WEBKIT_TEXT_COMBINER_PAD(pad);

    if (combinerPad->priv->shouldProcessStickyEvents) {
        gst_pad_sticky_events_foreach(pad, [](GstPad* pad, GstEvent** event, gpointer) -> gboolean {
            if (GST_EVENT_TYPE(*event) != GST_EVENT_CAPS)
                return TRUE;

            auto* combinerPad = WEBKIT_TEXT_COMBINER_PAD(pad);
            auto parent = adoptGRef(gst_pad_get_parent(pad));
            GstCaps* caps;
            gst_event_parse_caps(*event, &caps);
            combinerPad->priv->shouldProcessStickyEvents = false;
            webKitTextCombinerHandleCaps(WEBKIT_TEXT_COMBINER(parent.get()), pad, caps);
            return FALSE;
        }, nullptr);
    }

    return gst_proxy_pad_chain_default(pad, parent, buffer);
}

static void webkitTextCombinerPadGetProperty(GObject* object, unsigned propertyId, GValue* value, GParamSpec* pspec)
{
    auto* pad = WEBKIT_TEXT_COMBINER_PAD(object);
    switch (propertyId) {
    case PROP_PAD_TAGS: {
        auto locker = GstObjectLocker(object);
        if (pad->priv->tags)
            g_value_take_boxed(value, gst_tag_list_copy(pad->priv->tags.get()));
        break;
    }
    case PROP_INNER_COMBINER_PAD: {
        auto locker = GstObjectLocker(object);
        g_value_set_object(value, pad->priv->innerCombinerPad.get());
        break;
    }
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, propertyId, pspec);
        break;
    }
}

static void webkitTextCombinerPadSetProperty(GObject* object, guint propertyId, const GValue* value, GParamSpec* pspec)
{
    auto* pad = WEBKIT_TEXT_COMBINER_PAD(object);
    switch (propertyId) {
    case PROP_INNER_COMBINER_PAD: {
        auto locker = GstObjectLocker(object);
        pad->priv->innerCombinerPad = adoptGRef(GST_PAD_CAST(g_value_get_object(value)));
        break;
    }
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, propertyId, pspec);
        break;
    }
}

static void webkitTextCombinerPadConstructed(GObject* object)
{
    G_OBJECT_CLASS(webkit_text_combiner_pad_parent_class)->constructed(object);
    gst_ghost_pad_construct(GST_GHOST_PAD(object));
    gst_pad_set_event_function(GST_PAD_CAST(object), webkitTextCombinerPadEvent);
    gst_pad_set_chain_function(GST_PAD_CAST(object), webkitTextCombinerPadChain);
}

static void webkit_text_combiner_pad_class_init(WebKitTextCombinerPadClass* klass)
{
    auto* gobjectClass = G_OBJECT_CLASS(klass);

    gobjectClass->constructed = webkitTextCombinerPadConstructed;
    gobjectClass->get_property = GST_DEBUG_FUNCPTR(webkitTextCombinerPadGetProperty);
    gobjectClass->set_property = GST_DEBUG_FUNCPTR(webkitTextCombinerPadSetProperty);

    sObjProperties[PROP_PAD_TAGS] =
        g_param_spec_boxed("tags", nullptr, nullptr, GST_TYPE_TAG_LIST,
            static_cast<GParamFlags>(G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));

    sObjProperties[PROP_INNER_COMBINER_PAD] =
        g_param_spec_object("inner-combiner-pad", nullptr, nullptr, GST_TYPE_PAD,
            static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

    g_object_class_install_properties(gobjectClass, N_PROPERTIES, sObjProperties.data());
}

GstPad* webKitTextCombinerPadLeakInternalPadRef(WebKitTextCombinerPad* pad)
{
    return pad->priv->innerCombinerPad.leakRef();
}

#endif
