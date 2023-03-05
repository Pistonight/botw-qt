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
#include "botwqt-frame.h"

namespace botwqt {

Frame::Frame(const cv::Mat& frame) {
    cv::resize(frame, m_frame, cv::Size(BOTW_QT_IMAGE_WIDTH, BOTW_QT_IMAGE_HEIGHT), cv::INTER_AREA);
}

bool Frame::has_valid_score() const {
    int white_pixels = 0;
    for (int i = 0; i < m_frame.rows; i++) {
        for (int j = 0; j < m_frame.cols; j++) {
            if (m_frame.at<uint8_t>(i, j) != 0) {
                white_pixels++;
            }
        }
    }

    double ratio = 1 - (double(white_pixels) / (m_frame.rows * m_frame.cols));

    return ratio >= BOTW_QT_SCORE_MIN && ratio <= BOTW_QT_SCORE_MAX;
}

std::string Frame::to_hex() {
    if (!m_hex_cache.empty()) {
        return m_hex_cache;
    }
    constexpr int size = BOTW_QT_IMAGE_WIDTH*BOTW_QT_IMAGE_HEIGHT/8*2;
    char hex[size+1];
    hex[size] = '\0';
    int idx = 0;
    uint8_t byte = 0;
    int bit = 0;

    for (int i = 0; i < m_frame.rows; i++) {
        for (int j = 0; j < m_frame.cols; j++) {
            uint8_t new_bit = (m_frame.at<uint8_t>(i, j) > 128) ? 1 : 0;
            byte |= new_bit << (7 - bit);
            if (bit == 7) {
                sprintf(hex+idx, "%02x", byte);
                idx += 2;
                byte = 0;
                bit = 0;
            } else {
                bit++;
            }
        }
    }

    m_hex_cache = hex;
    return m_hex_cache;
}
}