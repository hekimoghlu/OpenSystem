/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 20, 2023.
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
#include "WebDateTimePickerGtk.h"

#include "WebKitWebViewBasePrivate.h"
#include <gtk/gtk.h>
#include <wtf/SetForScope.h>
#include <wtf/glib/GRefPtr.h>
#include <wtf/text/MakeString.h>

namespace WebKit {

Ref<WebDateTimePickerGtk> WebDateTimePickerGtk::create(WebPageProxy& page)
{
    return adoptRef(*new WebDateTimePickerGtk(page));
}

WebDateTimePickerGtk::~WebDateTimePickerGtk()
{
    invalidate();
}

WebDateTimePickerGtk::WebDateTimePickerGtk(WebPageProxy& page)
    : WebDateTimePicker(page)
{
}

void WebDateTimePickerGtk::invalidate()
{
    if (!m_popover)
        return;

    g_signal_handlers_disconnect_by_data(m_popover, this);
#if USE(GTK4)
    auto* webView = gtk_widget_get_parent(m_popover);
    gtk_widget_unparent(m_popover);
#else
    auto* webView = gtk_popover_get_relative_to(GTK_POPOVER(m_popover));
    gtk_widget_destroy(m_popover);
#endif
    m_popover = nullptr;
    m_calendar = nullptr;

    webkitWebViewBaseSetShouldNotifyFocusEvents(WEBKIT_WEB_VIEW_BASE(webView), true);
}

void WebDateTimePickerGtk::endPicker()
{
    invalidate();
    WebDateTimePicker::endPicker();
}

static String timeToString(const WebCore::DateComponents& time, WebCore::SecondFormat secondFormat)
{
    switch (secondFormat) {
    case WebCore::SecondFormat::None:
        return makeString(pad('0', 2, time.hour()), ':', pad('0', 2, time.minute()));
    case WebCore::SecondFormat::Second:
        return makeString(pad('0', 2, time.hour()), ':', pad('0', 2, time.minute()), ':', pad('0', 2, time.second()));
    case WebCore::SecondFormat::Millisecond:
        return makeString(pad('0', 2, time.hour()), ':', pad('0', 2, time.minute()), ':', pad('0', 2, time.second()), '.', pad('0', 3, time.millisecond()));
    }

    ASSERT_NOT_REACHED();
    return { };
}

static String calendarDateToString(int year, int month, int day, const std::optional<WebCore::DateComponents>& date, WebCore::SecondFormat secondFormat)
{
    auto type = date ? date->type() : WebCore::DateComponentsType::Date;
    switch (type) {
    case WebCore::DateComponentsType::Date:
        return makeString(pad('0', 4, year), '-', pad('0', 2, month + 1), '-', pad('0', 2, day));
    case WebCore::DateComponentsType::DateTimeLocal:
        return makeString(pad('0', 4, year), '-', pad('0', 2, month + 1), '-', pad('0', 2, day), 'T', timeToString(*date, secondFormat));
    case WebCore::DateComponentsType::Invalid:
    case WebCore::DateComponentsType::Month:
    case WebCore::DateComponentsType::Time:
    case WebCore::DateComponentsType::Week:
        break;
    }

    ASSERT_NOT_REACHED();
    return { };
}

void WebDateTimePickerGtk::didChooseDate()
{
    if (m_inUpdate)
        return;

    if (!m_page)
        return;

    int year, month, day;
    g_object_get(m_calendar, "year", &year, "month", &month, "day", &day, nullptr);
    m_page->didChooseDate(calendarDateToString(year, month, day, m_currentDate, m_secondFormat));
}

void WebDateTimePickerGtk::showDateTimePicker(WebCore::DateTimeChooserParameters&& params)
{
    if (m_popover) {
        update(WTFMove(params));
        return;
    }

    auto* webView = m_page->viewWidget();
    webkitWebViewBaseSetShouldNotifyFocusEvents(WEBKIT_WEB_VIEW_BASE(webView), false);

#if USE(GTK4)
    m_popover = gtk_popover_new();
    gtk_popover_set_has_arrow(GTK_POPOVER(m_popover), FALSE);
    gtk_widget_set_parent(m_popover, webView);
#else
    m_popover = gtk_popover_new(webView);
#endif
    gtk_popover_set_position(GTK_POPOVER(m_popover), GTK_POS_BOTTOM);
    GdkRectangle rectInRootView = params.anchorRectInRootView;
    gtk_popover_set_pointing_to(GTK_POPOVER(m_popover), &rectInRootView);
    g_signal_connect_swapped(m_popover, "closed", G_CALLBACK(+[](WebDateTimePickerGtk* picker) {
        picker->endPicker();
    }), this);

    m_calendar = gtk_calendar_new();
    g_signal_connect(m_calendar, "day-selected", G_CALLBACK(+[](GtkCalendar* calendar, WebDateTimePickerGtk* picker) {
        picker->didChooseDate();
    }), this);
#if USE(GTK4)
    gtk_popover_set_child(GTK_POPOVER(m_popover), m_calendar);
#else
    gtk_container_add(GTK_CONTAINER(m_popover), m_calendar);
    gtk_widget_show(m_calendar);
#endif

    update(WTFMove(params));

    gtk_popover_popup(GTK_POPOVER(m_popover));
}

void WebDateTimePickerGtk::update(WebCore::DateTimeChooserParameters&& params)
{
    SetForScope inUpdate(m_inUpdate, true);
    if (params.type == "date"_s)
        m_currentDate = WebCore::DateComponents::fromParsingDate(params.currentValue);
    else if (params.type == "datetime-local"_s)
        m_currentDate = WebCore::DateComponents::fromParsingDateTimeLocal(params.currentValue);

    if (m_currentDate)
        g_object_set(m_calendar, "year", m_currentDate->fullYear(), "month", m_currentDate->month(), "day", m_currentDate->monthDay(), nullptr);
    else if (params.type == "datetime-local"_s) {
        GRefPtr<GDateTime> now = adoptGRef(g_date_time_new_now_local());
        Seconds unixTime = Seconds(g_date_time_to_unix(now.get())) + Seconds::fromMicroseconds(g_date_time_get_utc_offset(now.get()));
        m_currentDate = WebCore::DateComponents::fromMillisecondsSinceEpochForDateTimeLocal(unixTime.milliseconds());
        if (params.hasMillisecondField)
            m_secondFormat = WebCore::SecondFormat::Millisecond;
        else if (params.hasSecondField)
            m_secondFormat = WebCore::SecondFormat::Second;
        else
            m_secondFormat = WebCore::SecondFormat::None;
    }
}

} // namespace WebKit
