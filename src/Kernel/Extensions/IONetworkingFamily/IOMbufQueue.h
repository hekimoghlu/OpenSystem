/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 7, 2023.
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
#ifndef _IOMBUFQUEUE_H
#define _IOMBUFQUEUE_H

extern "C" {
#include <sys/param.h>
#include <sys/mbuf.h>
}

struct IOMbufQueue
{
    mbuf_t      head;
    mbuf_t      tail;
    uint32_t    count;
    uint32_t    capacity;
    uint32_t    bytes;
};

static __inline__
int IOMbufFree(mbuf_t m)
{
	return mbuf_freem_list(m);
}

static __inline__
void IOMbufQueueInit( IOMbufQueue * q, uint32_t capacity = 0 )
{
    q->head  = q->tail = 0;
    q->count = 0;
    q->bytes = 0;
    q->capacity = capacity;
}

static __inline__
bool IOMbufQueueEnqueue( IOMbufQueue * q, mbuf_t m )
{
    if (q->count >= q->capacity)
        return false;

    if (q->count++ > 0)
        mbuf_setnextpkt(q->tail , m);
    else
        q->head = m;

    for (q->tail = m;
         mbuf_nextpkt(q->tail);
         q->tail = mbuf_nextpkt(q->tail), q->count++)
        ;

    return true;
}

static __inline__
bool IOMbufQueueEnqueue( IOMbufQueue * q, IOMbufQueue * qe )
{
    if (qe->count)
    {
        if (q->count == 0)
            q->head = qe->head;
        else
            mbuf_setnextpkt(q->tail , qe->head);
        q->tail = qe->tail;
        q->count += qe->count;

        qe->head = qe->tail = 0;
        qe->count = 0;
    }
    return true;
}

static __inline__
void IOMbufQueuePrepend( IOMbufQueue * q, mbuf_t m )
{
    mbuf_t tail;

    for (tail = m, q->count++;
         mbuf_nextpkt(tail);
         tail = mbuf_nextpkt(tail), q->count++)
        ;

    mbuf_setnextpkt(tail , q->head);
    if (q->tail == 0)
        q->tail = tail;
    q->head = m;
}

static __inline__
void IOMbufQueuePrepend( IOMbufQueue * q, IOMbufQueue * qp )
{
    if (qp->count)
    {
        mbuf_setnextpkt(qp->tail , q->head);
        if (q->tail == 0)
            q->tail = qp->tail;
        q->head = qp->head;
        q->count += qp->count;

        qp->head = qp->tail = 0;
        qp->count = 0;
    }
}

static __inline__
mbuf_t IOMbufQueueDequeue( IOMbufQueue * q )
{   
    mbuf_t m = q->head;
    if (m)
    {
        if ((q->head = mbuf_nextpkt(m)) == 0)
            q->tail = 0;
        mbuf_setnextpkt(m , 0);
        q->count--;
    }
    return m;
}

static __inline__
mbuf_t IOMbufQueueDequeueAll( IOMbufQueue * q )
{
    mbuf_t m = q->head;
    q->head = q->tail = 0;
    q->count = 0;
    return m;
}

static __inline__
mbuf_t IOMbufQueuePeek( IOMbufQueue * q )
{
    return q->head;
}

static __inline__
uint32_t IOMbufQueueGetSize( IOMbufQueue * q )
{
    return q->count;
}

static __inline__
uint32_t IOMbufQueueIsEmpty( IOMbufQueue * q )
{
    return (0 == q->count);
}

static __inline__
uint32_t IOMbufQueueGetCapacity( IOMbufQueue * q )
{
    return q->capacity;
}

static __inline__
void IOMbufQueueSetCapacity( IOMbufQueue * q, uint32_t capacity )
{
	q->capacity = capacity;
}

static __inline__
void IOMbufQueueTailAdd( IOMbufQueue * q, mbuf_t m, uint32_t len )
{
    if (q->count == 0)
    {
        q->head = q->tail = m;
    }
    else
    {
        mbuf_setnextpkt(q->tail, m);
        q->tail = m;
    }
    q->count++;
    q->bytes += len;
}

#endif /* !_IOMBUFQUEUE_H */
