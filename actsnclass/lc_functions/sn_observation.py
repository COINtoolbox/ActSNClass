import os
from actsnclass.data_functions.read_SNANA import read_snana_lc
import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D
import collections

try:
    from threeML import *

    no_threeml = False
    from threeML.io.file_utils import file_existing_and_readable

except:

    no_threeml = True

try:
    import pygmo
    no_pygmo = False
except ImportError:

    no_pygmo = True

no_threeml = False
no_pygmo = False

class SNObservation(object):
    def __init__(self, data_file, snr_table=None):

        header, flux = read_snana_lc(data_file, flux_as_DataFrame=True)
        self._sample = header['sample']

        self._pkmjd = header['pkmjd']

        self._pkmag = {}
        for band, val in zip(['g', 'i', 'r', 'z'], header['pkmag']):
            self._pkmag[band] = val

        self._type = header['type'][0]
        if self._type != 0:
            self._is_1a = False
        else:
            self._is_1a = True
        self._snid = header['snid']
        self._z = header['z']
        self._model = header['model']
        self._bands = collections.OrderedDict()

        if snr_table is not None:

            head, tail = os.path.split(data_file)

            use_band = snr_table[snr_table['sn_file'] == tail][0]

        else:

            use_band = {'g': True, 'r': True, 'i': True, 'z': True}

        for band, flux_table in flux.iteritems():

            if use_band[band]:

                sn_band = SNBand(flux_table, band, self)
                self._bands[band] = sn_band

                setattr(self, band, sn_band)

    def __repr__(self):

        s = ''

        s += 'snid: %d\n' % self._snid
        s += 'sample: %s\n' % self._sample
        s += 'is Ia: %s\n' % self._is_1a
        s += 'type: %s\n' % self._type
        s += 'bands: %s' % ' ,'.join(self._bands)
        return s

    @property
    def sample(self):
        return self._sample

    @property
    def pkmag(self):

        return self._pkmag

    @property
    def type(self):

        return self._type

    @property
    def is_1a(self):
        return self._is_1a

    @property
    def redshift(self):
        return self._z

    @property
    def snid(self):
        return self._snid

    @property
    def pkmjd(self):

        return self._pkmjd

    @property
    def model(self):

        return self._model

    def plot(self):
        """
        plot the light curves
        """

        fig, ax = plt.subplots()

        for band in self._bands.itervalues():

            _ = band.plot(ax=ax)

        ax.legend()

        return fig

    def fit(self,
            function,
            with_global_optimization=True,
            bands=None,
            quiet=False):

        if bands is None:
            bands = self._bands.keys()

        for band in bands:

            new_func = copy.deepcopy(function)

            self._bands[band].fit(new_func,
                                  with_global_optimization,
                                  quiet=quiet)

    def fit_multinest(self,
                      function,
                      bands=None,
                      quiet=False,
                      resume=False,
                      chain_name='',
                      check_fit=True,
                      destination=None,
                      info=None):

        if bands is None:
            bands = self._bands.keys()

        for band in bands:

            dontgo = True
            if check_fit:

                dontgo = self._bands[band].check_if_fit_exists(destination,
                                                               info)

            else:

                dontgo = False

            if not dontgo:

                new_func = copy.deepcopy(function)

                self._bands[band].fit_multinest(
                    new_func,
                    quiet=quiet,
                    resume=resume,
                    chain_name="%s_%s-" % (chain_name, band))

    def plot_fit(self):

        fig, ax = plt.subplots()

        for v in self._bands.itervalues():

            try:

                _ = v.plot_fit(ax)

            except:

                pass

        return fig

    def save_fit(self, destination='.', fit_info='', overwrite=False):

        for v in self._bands.itervalues():

            v.save_fit(destination, fit_info, overwrite)

    @property
    def bands(self):

        return self._bands

    @property
    def peak_flux(self):

        peak_flux = {}

        for k, v in self._bands.iteritems():

            peak_flux[k] = v.peak_flux

        return peak_flux

    @property
    def peak_time(self):

        peak_time = {}

        for k, v in self._bands.iteritems():

            peak_time[k] = v.peak_time

        return peak_time


class SNBand(object):

    _color_wheel = {
        'r': '#00ff7b',
        'g': '#00e1ff',
        'i': '#ff0000',
        'z': '#FCFF31'
        #        'z': '#ff7b00'
    }

    def __init__(self, flux_table, band, observation):

        self._observation = observation
        self._snid = observation.snid

        self._band = band

        err_idx = np.isfinite(np.array(flux_table['FLUXCALERR']))

        self._error = np.array(flux_table['FLUXCALERR'])[err_idx]

        self._mjd = np.array(flux_table['MJD'])[err_idx]
        self._flux = np.array(flux_table['FLUXCAL'])[err_idx]
        self._error = np.array(flux_table['FLUXCALERR'])[err_idx]
        self._snr = np.array(flux_table['SNR'])[err_idx]

        self._zero_time = self._mjd.min()

        self._idx_max = np.argmax(self._flux)

        self._normalized_time = self._mjd - self._zero_time

        self._jl = None

    @property
    def mjd(self):
        return self._mjd

    @property
    def normalized_time(self):

        return self._normalized_time

    @property
    def flux(self):
        return self._flux

    @property
    def error(self):
        return self._error

    @property
    def peak_time(self):

        return self._normalized_time[self._idx_max]

    @property
    def peak_flux(self):

        return self._flux[self._idx_max]

    @property
    def zero_time(self):

        return self._zero_time

    def plot(self, ax=None, norm_time=False):

        if ax is None:

            fig, ax = plt.subplots()

        else:

            fig = ax.get_figure()

        if norm_time:
            time = self._normalized_time

        else:

            time = self._mjd

        ax.errorbar(
            time,
            self._flux,
            yerr=self._error,
            color=self._color_wheel[self._band],
            fmt='o',
            label=self._band)

        return fig

    def fit_multinest(self, function, resume=False, chain_name='',
                      quiet=False):

        self._ps = PointSource(self._band, 0, 0, spectral_shape=function)
        self._model = Model(self._ps)

        self._xyl = XYLike('band_%s' % (self._band), self._normalized_time,
                           self._flux, self._error)
        self._jl = BayesianAnalysis(self._model, DataList(self._xyl))

        self._jl.sample_multinest(
            n_live_points=300,
            quiet=quiet,
            chain_name=chain_name,
            resume=resume,
            importance_nested_sampling=False)

    def fit(self, function, with_global_optimization=True, quiet=False):

        if no_threeml:
            raise RuntimeError('No 3ML!')

        self._ps = PointSource(self._band, 0, 0, spectral_shape=function)
        self._model = Model(self._ps)

        self._xyl = XYLike('band_%s' % (self._band), self._normalized_time,
                           self._flux, self._error)
        self._jl = JointLikelihood(self._model, DataList(self._xyl))

        if with_global_optimization:
            pagmo_minimizer = GlobalMinimization("pagmo")

            my_algorithm = pygmo.algorithm(pygmo.de(gen=25))

            # Create an instance of a local minimizer
            local_minimizer = LocalMinimization("ROOT")

            # Setup the global minimization
            pagmo_minimizer.setup(
                second_minimization=local_minimizer,
                algorithm=my_algorithm,
                islands=10,
                population_size=15,
                evolution_cycles=5)

            self._jl.set_minimizer(pagmo_minimizer)

        else:

            self._jl.set_minimizer('ROOT')

        try:

            _ = self._jl.fit(quiet=quiet)

        except:

            print('Fit FAILED')

            self._jl = None

    def check_if_fit_exists(self, destination='.', save_info=''):

        file_name = "%s/%s_%s_%s.fits" % (destination, self._band, self._snid,
                                          save_info)

        return file_existing_and_readable(file_name)

    def save_fit(self, destination='.', save_info='', overwrite=True):

        if self._jl is not None:

            file_name = "%s/%s_%s_%s.fits" % (destination, self._band,
                                              self._snid, save_info)

            self._jl.results.write_to(file_name, overwrite=overwrite)

    def plot_fit(self, ax=None):

        if self._jl is not None:

            return self._xyl.plot(
                x_label='time',
                y_label='flux',
                subplot=ax,
                model_color='k',
                data_color=self._color_wheel[self._band])

        else:
            print 'T11111'


data_path = os.path.join('..', 'data', 'SIMGEN_PUBLIC_DES')


class FittedObservation(object):
    def __init__(self, *fitted_bands, **kwargs):
        """
        :param fitted_bands: filenames for each band of this fit
        :param data_dir: the location of the DES data files
        """

        self._bands = {}
        tmp_snids = []
        tmp_functions = []

        for f in fitted_bands:

            filename = f.split('/')[-1]

            band, snid, ext = filename.split('_')

            function, _ = ext.split('.')

            ar = load_analysis_results(f)

            self._bands[band] = ar

            tmp_snids.append(snid)
            tmp_functions.append(function)

            setattr(self, band, ar)

        assert len(np.unique(tmp_snids)) == 1
        assert len(np.unique(tmp_functions)) == 1

        if 'data_dir' in kwargs:

            data_dir = kwargs.pop('data_dir')

            snid = tmp_snids[0]

            snid_str = snid.zfill(6)

            data_file = os.path.join(data_dir, 'DES_SN%s.DAT' % snid_str)
            self._snobs = SNObservation(data_file)

        else:

            self._snobs = None

    def extract_credible_regions(self, time_range, band):

        assert band in self._bands.keys()

        ps = self._bands[band].optimized_model.point_sources[band]

        parameters = ps.spectrum.main.shape.parameters
        test_model = ps.spectrum.main.shape
        parameter_names = [
            par.name for par in ps.spectrum.main.shape.parameters.values()
        ]

        arguments = {}

        # because we might be using composite functions,
        # we have to keep track of parameter names in a non-elegant way
        for par, name in zip(parameters.values(), parameter_names):

            if par.free:

                this_variate = self._bands[band].get_variates(par.path)

                # Do not use more than 1000 values (would make computation too slow for nothing)

                if len(this_variate) > 1000:
                    this_variate = np.random.choice(this_variate, size=1000)

                    arguments[name] = this_variate

            else:

                # use the fixed value rather than a variate

                arguments[name] = par.value

                # create the propagtor

        propagated_function = self._bands[band].propagate(test_model.evaluate,
                                                          **arguments)

        return [propagated_function(t) for t in time_range]

    def posterior_plot(self, time_range, thin=10, with_data=True):

        trace_colors = {
            'r': '#076d00',
            'g': '#008b9b',
            'i': '#a50018',
            'z': '#9b4800'
        }

        fig, ax = plt.subplots()

        for band in self._bands:

            # plot the data
            if with_data:
                self._snobs.bands[band].plot(ax, norm_time=True)

            # now build the samples range
            variates_in_time = self.extract_credible_regions(time_range, band)
            samples_in_time = np.array(
                [v.samples[::thin].tolist() for v in variates_in_time]).T

            for sample in samples_in_time:

                if with_data:

                    ax.plot(
                        time_range, sample, color=trace_colors[band], alpha=.1)

                else:
                    ax.plot(
                        time_range,
                        sample,
                        color=SNBand._color_wheel[band],
                        alpha=.1)

        ax.legend()
        return fig

    def posterior_plot_3D(self,
                          time_range,
                          thin=10,
                          with_data=False,
                          with_cloud=True,
                          n_cloud_points=20,
                          point_size=20,
                          point_color='k',axis_color='k',bkg_color=None):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for k, band in enumerate(self._bands):

            # now build the samples range
            variates_in_time = self.extract_credible_regions(time_range, band)
            samples_in_time = np.array(
                [v.samples[::thin].tolist() for v in variates_in_time]).T

            if with_data:

                xs = self._snobs.bands[band].normalized_time
                ys = self._snobs.bands[band].flux
                err = self._snobs.bands[band].error
                if not with_cloud:
                    ax.scatter(
                        [k + 1] * len(xs), xs, ys, c=point_color, s=point_size)
                    for x,y,e in zip(xs,ys,err):

                        ax.plot([k+1,k+1],[x,x],[y-e,y+e], lw=1, color=point_color)
                    
                else:

                    for x, y, e in zip(xs, ys, err):

                        new_k = norm.rvs(loc=[k + 1],
                                         scale=.1,
                                         size=n_cloud_points)
                        new_x = norm.rvs(loc=x, scale=.1, size=n_cloud_points)
                        new_y = norm.rvs(loc=y, scale=e, size=n_cloud_points)

                        for a, b, c in zip(new_k, new_x, new_y):
                            alp1 = norm.pdf(c, loc=y, scale=e)
                            
                            ax.scatter(
                                a,
                                b,
                                c,
                                c=point_color,
                                s=point_size,
                                alpha=alp1)

            for sample in samples_in_time:

                xx = norm.rvs(loc=k + 1, scale=.1)

                alpha = .05 * xx / float(k + 1)  #norm.rvs(loc=.2,scale=.1)

                ax.plot(
                    [xx] * len(time_range),
                    time_range,
                    sample,
                    lw=2,
                    color=SNBand._color_wheel[band],
                    alpha=alpha)

        ax.set_ylabel('Time')
        ax.set_zlabel('Flux')

        ax.grid(False)

        ax.xaxis.pane.set_edgecolor(axis_color)
        ax.yaxis.pane.set_edgecolor(axis_color)
        if bkg_color is not None:
            ax.xaxis.pane.set_color(bkg_color)
            ax.yaxis.pane.set_color(bkg_color)
            ax.zaxis.pane.set_color(bkg_color)

        else:
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False

        ax.set_zlim(0.)
        ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.set_zticks([])

        plt.xticks(np.arange(len(self._bands)) + 1, self._bands)

    @property
    def sn_obs(self):

        return self._snobs

    def plot(self):

        pass
