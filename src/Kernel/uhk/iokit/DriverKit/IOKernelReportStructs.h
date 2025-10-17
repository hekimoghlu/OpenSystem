/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 11, 2025.
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
// Internal data structures to be used by IOReporters and User Space Observers


#ifndef _IOKERNELREPORTSTRUCTS_H_
#define _IOKERNELREPORTSTRUCTS_H_

#include <stdint.h>

#if KERNEL
#include <IOKit/IOReportTypes.h>
#else
#include <DriverKit/IOReportTypes.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Drivers participating in IOReporting can advertise channels by
// publishing properties in the I/O Kit registry.  Various helper
// mechanisms exist to produce correctly-formatted legends.
// 12836893 tracks advertising channels in user space.
#define kIOReportLegendPublicKey        "IOReportLegendPublic"      // bool
#define kIOReportLegendKey              "IOReportLegend"            // arr
#define kIOReportLegendChannelsKey      "IOReportChannels"          // arr
#define kIOReportLegendGroupNameKey     "IOReportGroupName"         // str
#define kIOReportLegendSubGroupNameKey  "IOReportSubGroupName"      // str
#define kIOReportLegendInfoKey          "IOReportChannelInfo"       // dict
#define kIOReportLegendUnitKey          "IOReportChannelUnit"       // num
#define kIOReportLegendConfigKey        "IOReportChannelConfig"     // data
#define kIOReportLegendStateNamesKey    "IOReportChannelStateNames" // str[]

// in an I/O Kit registry legend, a small "array struct" represents a channel
#define kIOReportChannelIDIdx           0       // required
#define kIOReportChannelTypeIdx         1       // required
#define kIOReportChannelNameIdx         2       // optional

/*  Histogram Segment Configuration
 *   Currently supports 2 types of scaling to compute bucket upper bounds,
 *   linear or exponential.
 *   scale_flag = 0 -> linear scale
 *                1 -> exponential scale
 *   upper_bound[n] = (scale_flag) ? pow(base,(n+1)) : base * (n+1);
 */
#define kIOHistogramScaleLinear 0
#define kIOHistogramScaleExponential 1
typedef struct {
	uint32_t    base_bucket_width;// segment[0].bucket[0] = [0, base_width]
	uint32_t    scale_flag;       // bit 0 only in current use (see #defs)
	uint32_t    segment_idx;      // for multiple segments histograms
	uint32_t    segment_bucket_count;// number of buckets in this segment
} __attribute((packed)) IOHistogramSegmentConfig;

// "normalized distribution"(FIXME?) internal format (unused?)
typedef struct {
	uint64_t    samples;
	uint64_t    mean;
	uint64_t    variance;
	uint64_t    reserved;
} __attribute((packed)) IONormDistReportValues;

#ifdef __cplusplus
}
#endif

#endif // _IOKERNELREPORTSTRUCTS_H_
