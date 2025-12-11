#ifndef PARSER_HPP
#define PARSER_HPP

#include "common.hpp"
#include <algorithm>
#include <sstream>

static inline std::string &ltrim(std::string &s)
{
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch)
                                    { return !std::isspace(ch); }));
    return s;
}

static inline std::string &rtrim(std::string &s)
{
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch)
                         { return !std::isspace(ch); })
                .base(),
            s.end());
    return s;
}

static inline std::string &trim(std::string &s)
{
    return ltrim(rtrim(s));
}

static double parse_angle(std::string s)
{
    trim(s);
    if (s.empty())
        return 0.0;

    size_t pi_pos = s.find("pi");
    if (pi_pos != std::string::npos)
    {
        std::string before_pi = s.substr(0, pi_pos);
        std::string after_pi = s.substr(pi_pos + 2);
        trim(before_pi);
        trim(after_pi);

        double multiplier = 1.0;
        double divisor = 1.0;

        if (!before_pi.empty())
        {
            if (before_pi == "+")
                multiplier = 1.0;
            else if (before_pi == "-")
                multiplier = -1.0;
            else
            {
                if (!before_pi.empty() && before_pi.back() == '*')
                    before_pi.pop_back();
                trim(before_pi);
                try
                {
                    multiplier = std::stod(before_pi);
                }
                catch (...)
                {
                    multiplier = 1.0;
                }
            }
        }

        if (!after_pi.empty() && after_pi.rfind('/', 0) == 0)
        {
            std::string denom_str = after_pi.substr(1);
            trim(denom_str);
            try
            {
                divisor = std::stod(denom_str);
                if (divisor == 0.0)
                    divisor = 1.0;
            }
            catch (...)
            {
                divisor = 1.0;
            }
        }
        return multiplier * M_PI / divisor;
    }

    try
    {
        return std::stod(s);
    }
    catch (...)
    {
        return 0.0;
    }
}

#endif // PARSER_HPP