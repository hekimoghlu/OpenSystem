/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 27, 2023.
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
#include "SampleMap.h"

#include "MediaSample.h"

namespace WebCore {

template <typename M>
class SampleIsLessThanMediaTimeComparator {
public:
    typedef typename M::value_type value_type;
    bool operator()(const value_type& value, const MediaTime& time)
    {
        MediaTime presentationEndTime = value.second->presentationTime() + value.second->duration();
        return presentationEndTime <= time;
    }
    bool operator()(const MediaTime& time, const value_type& value)
    {
        MediaTime presentationStartTime = value.second->presentationTime();
        return time < presentationStartTime;
    }
};

template <typename M>
class SampleIsGreaterThanMediaTimeComparator {
public:
    typedef typename M::value_type value_type;
    bool operator()(const value_type& value, const MediaTime& time)
    {
        MediaTime presentationStartTime = value.second->presentationTime();
        return presentationStartTime > time;
    }
    bool operator()(const MediaTime& time, const value_type& value)
    {
        MediaTime presentationEndTime = value.second->presentationTime() + value.second->duration();
        return time >= presentationEndTime;
    }
};

class SampleIsRandomAccess {
public:
    bool operator()(DecodeOrderSampleMap::MapType::value_type& value)
    {
        return value.second->isSync();
    }
};

// SamplePresentationTimeIsInsideRangeComparator matches (range.first, range.second]
struct SamplePresentationTimeIsInsideRangeComparator {
    bool operator()(std::pair<MediaTime, MediaTime> range, const std::pair<MediaTime, RefPtr<MediaSample>>& value)
    {
        return range.second < value.first;
    }
    bool operator()(const std::pair<MediaTime, RefPtr<MediaSample>>& value, std::pair<MediaTime, MediaTime> range)
    {
        return value.first <= range.first;
    }
};

// SamplePresentationTimeIsWithinRangeComparator matches [range.first, range.second)
struct SamplePresentationTimeIsWithinRangeComparator {
    bool operator()(std::pair<MediaTime, MediaTime> range, const std::pair<MediaTime, RefPtr<MediaSample>>& value)
    {
        return range.second <= value.first;
    }
    bool operator()(const std::pair<MediaTime, RefPtr<MediaSample>>& value, std::pair<MediaTime, MediaTime> range)
    {
        return value.first < range.first;
    }
};

bool SampleMap::empty() const
{
    return presentationOrder().m_samples.empty();
}

void SampleMap::clear()
{
    presentationOrder().m_samples.clear();
    decodeOrder().m_samples.clear();
    m_totalSize = 0;
}

void SampleMap::addSample(Ref<MediaSample>&& sample)
{
    MediaTime presentationTime = sample->presentationTime();

    m_totalSize += sample->sizeInBytes();

    presentationOrder().m_samples.insert(PresentationOrderSampleMap::MapType::value_type(presentationTime, sample));

    auto decodeKey = DecodeOrderSampleMap::KeyType(sample->decodeTime(), presentationTime);
    decodeOrder().m_samples.insert(DecodeOrderSampleMap::MapType::value_type(decodeKey, WTFMove(sample)));
}

void SampleMap::removeSample(const MediaSample& sample)
{
    MediaTime presentationTime = sample.presentationTime();

    m_totalSize -= sample.sizeInBytes();

    auto decodeKey = DecodeOrderSampleMap::KeyType(sample.decodeTime(), presentationTime);
    presentationOrder().m_samples.erase(presentationTime);
    decodeOrder().m_samples.erase(decodeKey);
}

PresentationOrderSampleMap::iterator PresentationOrderSampleMap::findSampleWithPresentationTime(const MediaTime& time)
{
    auto range = m_samples.equal_range(time);
    if (range.first == range.second)
        return end();
    return range.first;
}

PresentationOrderSampleMap::iterator PresentationOrderSampleMap::findSampleContainingPresentationTime(const MediaTime& time)
{
    // upper_bound will return the first sample whose presentation start time is greater than the search time.
    // If this is the first sample, that means no sample in the map contains the requested time.
    auto iter = m_samples.upper_bound(time);
    if (iter == begin())
        return end();

    // Look at the previous sample; does it contain the requested time?
    --iter;
    const auto& sample = iter->second;
    if (sample->presentationTime() + sample->duration() > time)
        return iter;
    return end();
}

PresentationOrderSampleMap::iterator PresentationOrderSampleMap::findSampleContainingOrAfterPresentationTime(const MediaTime& time)
{
    if (m_samples.empty())
        return end();

    // upper_bound will return the first sample whose presentation start time is greater than the search time.
    // If this is the first sample, that means no sample in the map contains the requested time.
    auto iter = m_samples.upper_bound(time);
    if (iter == begin())
        return iter;

    // Look at the previous sample; does it contain the requested time?
    --iter;
    const auto& sample = iter->second;
    if (sample->presentationTime() + sample->duration() > time)
        return iter;
    return ++iter;
}

PresentationOrderSampleMap::iterator PresentationOrderSampleMap::findSampleStartingOnOrAfterPresentationTime(const MediaTime& time)
{
    return m_samples.lower_bound(time);
}

PresentationOrderSampleMap::iterator PresentationOrderSampleMap::findSampleStartingAfterPresentationTime(const MediaTime& time)
{
    return m_samples.upper_bound(time);
}

DecodeOrderSampleMap::iterator DecodeOrderSampleMap::findSampleWithDecodeKey(const KeyType& key)
{
    return m_samples.find(key);
}

DecodeOrderSampleMap::iterator DecodeOrderSampleMap::findSampleAfterDecodeKey(const KeyType& key)
{
    return m_samples.upper_bound(key);
}

PresentationOrderSampleMap::reverse_iterator PresentationOrderSampleMap::reverseFindSampleContainingPresentationTime(const MediaTime& time)
{
    auto range = std::equal_range(rbegin(), rend(), time, SampleIsGreaterThanMediaTimeComparator<MapType>());
    if (range.first == range.second)
        return rend();
    return range.first;
}

PresentationOrderSampleMap::reverse_iterator PresentationOrderSampleMap::reverseFindSampleBeforePresentationTime(const MediaTime& time)
{
    if (m_samples.empty())
        return rend();

    // upper_bound will return the first sample whose presentation start time is greater than the search time.
    auto found = m_samples.upper_bound(time);

    // If no sample was found with a time greater than the search time, return the last sample.
    if (found == end())
        return rbegin();

    // If the first sample has a time grater than the search time, no samples will have a presentation time before the search time.
    if (found == begin())
        return rend();

    // Otherwise, return the sample immediately previous to the one found.
    return --reverse_iterator(--found);
}

DecodeOrderSampleMap::reverse_iterator DecodeOrderSampleMap::reverseFindSampleWithDecodeKey(const KeyType& key)
{
    DecodeOrderSampleMap::iterator found = findSampleWithDecodeKey(key);
    if (found == end())
        return rend();
    return --reverse_iterator(found);
}

DecodeOrderSampleMap::reverse_iterator DecodeOrderSampleMap::findSyncSamplePriorToPresentationTime(const MediaTime& time, const MediaTime& threshold)
{
    PresentationOrderSampleMap::reverse_iterator reverseCurrentSamplePTS = m_presentationOrder.reverseFindSampleBeforePresentationTime(time);
    if (reverseCurrentSamplePTS == m_presentationOrder.rend())
        return rend();

    const auto& sample = reverseCurrentSamplePTS->second;
    reverse_iterator reverseCurrentSampleDTS = reverseFindSampleWithDecodeKey(KeyType(sample->decodeTime(), sample->presentationTime()));

    reverse_iterator foundSample = findSyncSamplePriorToDecodeIterator(reverseCurrentSampleDTS);
    if (foundSample == rend())
        return rend();
    if (foundSample->second->presentationTime() < time - threshold)
        return rend();
    return foundSample;
}

DecodeOrderSampleMap::reverse_iterator DecodeOrderSampleMap::findSyncSamplePriorToDecodeIterator(reverse_iterator iterator)
{
    return std::find_if(iterator, rend(), SampleIsRandomAccess());
}

DecodeOrderSampleMap::iterator DecodeOrderSampleMap::findSyncSampleAfterPresentationTime(const MediaTime& time, const MediaTime& threshold)
{
    PresentationOrderSampleMap::iterator currentSamplePTS = m_presentationOrder.findSampleStartingOnOrAfterPresentationTime(time);
    if (currentSamplePTS == m_presentationOrder.end())
        return end();

    const auto& sample = currentSamplePTS->second;
    iterator currentSampleDTS = findSampleWithDecodeKey(KeyType(sample->decodeTime(), sample->presentationTime()));
    
    MediaTime upperBound = time + threshold;
    iterator foundSample = std::find_if(currentSampleDTS, end(), SampleIsRandomAccess());
    if (foundSample == end())
        return end();
    if (foundSample->second->presentationTime() > upperBound)
        return end();
    return foundSample;
}

DecodeOrderSampleMap::iterator DecodeOrderSampleMap::findSyncSampleAfterDecodeIterator(iterator currentSampleDTS)
{
    if (currentSampleDTS == end())
        return end();
    return std::find_if(++currentSampleDTS, end(), SampleIsRandomAccess());
}

PresentationOrderSampleMap::iterator_range PresentationOrderSampleMap::findSamplesBetweenPresentationTimes(const MediaTime& beginTime, const MediaTime& endTime)
{
    // startTime is inclusive, so use lower_bound to include samples wich start exactly at startTime.
    // endTime is not inclusive, so use lower_bound to exclude samples which start exactly at endTime.
    auto lower_bound = m_samples.lower_bound(beginTime);
    auto upper_bound = m_samples.lower_bound(endTime);
    if (lower_bound == upper_bound)
        return { end(), end() };
    return { lower_bound, upper_bound };
}

PresentationOrderSampleMap::iterator_range PresentationOrderSampleMap::findSamplesBetweenPresentationTimesFromEnd(const MediaTime& beginTime, const MediaTime& endTime)
{
    reverse_iterator rangeEnd = std::find_if(rbegin(), rend(), [&endTime](const auto& value) {
        return value.first < endTime;
    });

    reverse_iterator rangeStart = std::find_if(rangeEnd, rend(), [&beginTime](const auto& value) {
        return value.first < beginTime;
    });

    if (rangeStart == rangeEnd)
        return { end(), end() };

    // ( rangeStart, rangeEnd ] == [ rangeStart.base(), rangeEnd.base() )
    return { rangeStart.base(), rangeEnd.base() };
}

DecodeOrderSampleMap::reverse_iterator_range DecodeOrderSampleMap::findDependentSamples(const MediaSample& sample)
{
    reverse_iterator currentDecodeIter = reverseFindSampleWithDecodeKey(KeyType(sample.decodeTime(), sample.presentationTime()));
    reverse_iterator nextSyncSample = findSyncSamplePriorToDecodeIterator(currentDecodeIter);
    return reverse_iterator_range(currentDecodeIter, nextSyncSample);
}

DecodeOrderSampleMap::iterator_range DecodeOrderSampleMap::findSamplesBetweenDecodeKeys(const KeyType& beginKey, const KeyType& endKey)
{
    if (beginKey > endKey)
        return { end(), end() };

    // beginKey is inclusive, so use lower_bound to include samples wich start exactly at beginKey.
    // endKey is not inclusive, so use lower_bound to exclude samples which start exactly at endKey.
    auto lower_bound = m_samples.lower_bound(beginKey);
    auto upper_bound = m_samples.lower_bound(endKey);
    if (lower_bound == upper_bound)
        return { end(), end() };
    ASSERT(lower_bound != end());
    if (lower_bound == end())
        return { begin(), upper_bound };
    return { lower_bound, upper_bound };
}

}
