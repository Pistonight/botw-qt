/*
botw-qt-backend
Copyright (C) 2023 iTNTPiston

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program. If not, see <https://www.gnu.org/licenses/>
*/

#ifndef _BOTWQT_FRAME_H_
#define _BOTWQT_FRAME_H_

#include <opencv2/opencv.hpp>

// The range of non-white pixel ratio that should be processed
#define BOTW_QT_SCORE_MIN 0.01
#define BOTW_QT_SCORE_MAX 0.45
#define BOTW_QT_IMAGE_WIDTH 492
#define BOTW_QT_IMAGE_HEIGHT 46

namespace botwqt {

// Wrapper for a quest banner image frame
class Frame {
public:
    Frame(const cv::Mat& frame);
    ~Frame() {}

    // Return if the frame has a valid black pixel ratio for it to be a banner
    bool has_valid_score() const;
    // Return the frame encoded as a hex string
    // Each byte represent 8 pixels, with bit 1 = white and bit 0 = black (big endian)
    // The bits are big-endian and in row-major order
    std::string to_hex();
private:
    cv::Mat m_frame;
    std::string m_hex_cache;
};
}

#endif
