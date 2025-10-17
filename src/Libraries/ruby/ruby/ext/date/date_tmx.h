/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 7, 2021.
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

#ifndef DATE_TMX_H
#define DATE_TMX_H

struct tmx_funcs {
    VALUE (*year)(void *dat);
    int (*yday)(void *dat);
    int (*mon)(void *dat);
    int (*mday)(void *dat);
    VALUE (*cwyear)(void *dat);
    int (*cweek)(void *dat);
    int (*cwday)(void *dat);
    int (*wnum0)(void *dat);
    int (*wnum1)(void *dat);
    int (*wday)(void *dat);
    int (*hour)(void *dat);
    int (*min)(void *dat);
    int (*sec)(void *dat);
    VALUE (*sec_fraction)(void *dat);
    VALUE (*secs)(void *dat);
    VALUE (*msecs)(void *dat);
    int (*offset)(void *dat);
    char *(*zone)(void *dat);
};
struct tmx {
    void *dat;
    const struct tmx_funcs *funcs;
};

#define tmx_attr(x) (tmx->funcs->x)(tmx->dat)

#define tmx_year tmx_attr(year)
#define tmx_yday tmx_attr(yday)
#define tmx_mon tmx_attr(mon)
#define tmx_mday tmx_attr(mday)
#define tmx_cwyear tmx_attr(cwyear)
#define tmx_cweek tmx_attr(cweek)
#define tmx_cwday tmx_attr(cwday)
#define tmx_wnum0 tmx_attr(wnum0)
#define tmx_wnum1 tmx_attr(wnum1)
#define tmx_wday tmx_attr(wday)
#define tmx_hour tmx_attr(hour)
#define tmx_min tmx_attr(min)
#define tmx_sec tmx_attr(sec)
#define tmx_sec_fraction tmx_attr(sec_fraction)
#define tmx_secs tmx_attr(secs)
#define tmx_msecs tmx_attr(msecs)
#define tmx_offset tmx_attr(offset)
#define tmx_zone tmx_attr(zone)

#endif

/*
Local variables:
c-file-style: "ruby"
End:
*/
