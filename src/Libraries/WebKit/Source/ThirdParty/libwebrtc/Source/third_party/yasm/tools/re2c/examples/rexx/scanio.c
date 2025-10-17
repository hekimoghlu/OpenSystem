/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 14, 2025.
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

uchar *ScanFill(uchar *cursor){
    unsigned cnt = s->tok - s->bot;
    s->pos += cursor - s->mrk;
    if(cnt){
        if(s->eot){
            unsigned len = s->eot - s->tok;
            memcpy(s->bot, s->tok, len);
            s->eot = &s->bot[len];
            if((len = s->lim - cursor) != 0)
                memcpy(s->eot, cursor, len);
            cursor = s->eot;
            s->lim = &cursor[len];
        } else {
            memcpy(s->bot, s->tok, s->lim - s->tok);
            cursor -= cnt;
            s->lim -= cnt;
        }
        s->tok = s->bot;
        s->ptr -= cnt;
    }
    if((s->top - s->lim) < 512){
        uchar *buf = (uchar*) malloc(((s->lim - s->bot) + 512)*sizeof(uchar));
        memcpy(buf, s->bot, s->lim - s->bot);
        s->tok = buf;
        s->ptr = &buf[s->ptr - s->bot];
        if(s->eot)
            s->eot = &buf[s->eot - s->bot];
        cursor = &buf[cursor - s->bot];
        s->lim = &buf[s->lim - s->bot];
        s->top = &s->lim[512];
        free(s->bot);
        s->bot = buf;
    }
    s->mrk = cursor;
    if(ScanCBIO.file){
        if((cnt = read(ScanCBIO.u.f.fd, (char*) s->lim, 512)) != 512)
            memset(&s->lim[cnt], 0, 512 - cnt);
        s->lim += 512;
    }
    return cursor;
}
