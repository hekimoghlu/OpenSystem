/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 24, 2022.
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
#include "PrintInfo.h"

#include <gtk/gtk.h>
#if HAVE(GTK_UNIX_PRINTING)
#include <gtk/gtkunixprint.h>
#include <WebCore/FloatConversion.h>
#endif

namespace WebKit {

#if HAVE(GTK_UNIX_PRINTING)
PrintInfo::PrintInfo(GtkPrintJob* job, PrintMode printMode)
    : printMode(printMode)
{
    ASSERT(job);

    GRefPtr<GtkPrintSettings> jobSettings;
    GRefPtr<GtkPageSetup> jobPageSetup;
    g_object_get(job, "settings", &jobSettings.outPtr(), "page-setup", &jobPageSetup.outPtr(), nullptr);

    pageSetupScaleFactor = gtk_print_settings_get_scale(jobSettings.get()) / 100.0;
    availablePaperWidth = gtk_page_setup_get_paper_width(jobPageSetup.get(), GTK_UNIT_POINTS) - gtk_page_setup_get_left_margin(jobPageSetup.get(), GTK_UNIT_POINTS) - gtk_page_setup_get_right_margin(jobPageSetup.get(), GTK_UNIT_POINTS);
    availablePaperHeight = gtk_page_setup_get_paper_height(jobPageSetup.get(), GTK_UNIT_POINTS) - gtk_page_setup_get_top_margin(jobPageSetup.get(), GTK_UNIT_POINTS) - gtk_page_setup_get_bottom_margin(jobPageSetup.get(), GTK_UNIT_POINTS);
    margin = { WebCore::narrowPrecisionToFloat(gtk_page_setup_get_top_margin(jobPageSetup.get(), GTK_UNIT_POINTS)), WebCore::narrowPrecisionToFloat(gtk_page_setup_get_right_margin(jobPageSetup.get(), GTK_UNIT_POINTS)), WebCore::narrowPrecisionToFloat(gtk_page_setup_get_bottom_margin(jobPageSetup.get(), GTK_UNIT_POINTS)), WebCore::narrowPrecisionToFloat(gtk_page_setup_get_left_margin(jobPageSetup.get(), GTK_UNIT_POINTS)) };

    pageSetup = WTFMove(jobPageSetup);

    printSettings = adoptGRef(gtk_print_settings_new());
    gtk_print_settings_set_printer_lpi(printSettings.get(), gtk_print_settings_get_printer_lpi(jobSettings.get()));

    if (const char* outputFormat = gtk_print_settings_get(jobSettings.get(), GTK_PRINT_SETTINGS_OUTPUT_FILE_FORMAT))
        gtk_print_settings_set(printSettings.get(), GTK_PRINT_SETTINGS_OUTPUT_FILE_FORMAT, outputFormat);
    else {
        auto* printer = gtk_print_job_get_printer(job);
        if (gtk_printer_accepts_pdf(printer))
            gtk_print_settings_set(printSettings.get(), GTK_PRINT_SETTINGS_OUTPUT_FILE_FORMAT, "pdf");
        else if (gtk_printer_accepts_ps(printer))
            gtk_print_settings_set(printSettings.get(), GTK_PRINT_SETTINGS_OUTPUT_FILE_FORMAT, "ps");
    }

    int rangesCount;
    auto* pageRanges = gtk_print_job_get_page_ranges(job, &rangesCount);
    gtk_print_settings_set_page_ranges(printSettings.get(), pageRanges, rangesCount);
    gtk_print_settings_set_print_pages(printSettings.get(), gtk_print_job_get_pages(job));
    gtk_print_settings_set_bool(printSettings.get(), "wk-rotate-to-orientation", gtk_print_job_get_rotate(job));
    gtk_print_settings_set_number_up(printSettings.get(), gtk_print_job_get_n_up(job));
    gtk_print_settings_set_number_up_layout(printSettings.get(), gtk_print_job_get_n_up_layout(job));
    gtk_print_settings_set_page_set(printSettings.get(), gtk_print_job_get_page_set(job));
    gtk_print_settings_set_reverse(printSettings.get(), gtk_print_job_get_reverse(job));
    gtk_print_settings_set_n_copies(printSettings.get(), gtk_print_job_get_num_copies(job));
    gtk_print_settings_set_collate(printSettings.get(), gtk_print_job_get_collate(job));
    gtk_print_settings_set_scale(printSettings.get(), gtk_print_job_get_scale(job));
}
#endif

} // namespace WebKit
