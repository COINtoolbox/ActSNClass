from light_curve_functions import karpenka, sanders, newling

try:
    from astromodels import Function1D, FunctionMeta, astropy_units

    class SN_Karpenka(Function1D):
        r"""
        description :
            SN light curve from Karpenka et al (2013)
        latex : $ A (1+ B(t-t_1)^2) \frac{\exp(-(t-t_0)/t_{\rm fall})}{1+\exp(-(t-t_0)/t_{\rm rise})} $
        parameters :
            A :
                desc : Normalization of first pulse
                initial value : 1.0
                is_normalization : True
                
                min : 0.
                max : 1e6
                delta : 0.1
            B :
                desc : Normalization of second pulse
                initial value : 1.0
                is_normalization : True
                
                min : 0.
                max : 1e3
                delta : 0.1
                
            t0 :
                desc : time normalization
                initial value : 0
                min : 0.
                max : 100
                
            t1 :
                desc : time normalization of second pulse
                initial value : 0
                min : 0
                max : 100
                
            t_fall :
                desc : time decay constant of pulse
                initial value : 1.
                min : 1.
                max : 100
                
            t_rise :
                desc : time rise constant of pulse
                initial value : 1.
                min : 1.
                max : 100
        
        """

        __metaclass__ = FunctionMeta

        def _set_units(self, x_unit, y_unit):
           
            
            self.t0.unit = x_unit
            self.t1.unit = x_unit
            self.t_fall.unit = x_unit
            self.t_rise.unit = x_unit

            # The normalization has the same units as the y

            self.A.unit = y_unit
            self.B.unit = y_unit

        # noinspection PyPep8Naming
        def evaluate(self, x, A, B, t0, t1, t_fall, t_rise):

            return karpenka(x, A, B, t0, t1, t_fall, t_rise)


    class SN_Bazin(Function1D):
        r"""
        description :
            SN light curve from Bazin et al (2013)
        latex : $ A \frac{\exp(-(t-t_0)/t_{\rm fall})}{1+\exp(-(t-t_0)/t_{\rm rise})} $
        parameters :
            A :
                desc : Normalization of first pulse
                initial value : 1.0
                is_normalization : True
                
                min : 0.
                max : 1e6
                delta : 0.1
            t0 :
                desc : time normalization
                initial value : 0
                min : 0.
                max : 200
                
                            
            t_fall :
                desc : time decay constant of pulse
                initial value : 1.
                min : 1.
                max : 100
                
            t_rise :
                desc : time rise constant of pulse
                initial value : 1.
                min : 1.
                max : 100
        
        """

        __metaclass__ = FunctionMeta

        def _set_units(self, x_unit, y_unit):
           
            
            self.t0.unit = x_unit

            self.t_fall.unit = x_unit
            self.t_rise.unit = x_unit

            # The normalization has the same units as the y

            self.A.unit = y_unit


        # noinspection PyPep8Naming
        def evaluate(self, x, A, t0, t_fall, t_rise):

            return karpenka(x, A, 0., t0, 0., t_fall, t_rise)


    class SN_Newling(Function1D):
        r"""
        description :
            SN light curve from Newling et al (2011)
        latex : $ A (1+ B(t-t_1)^2) \frac{\exp(-(t-t_0)/t_{\rm fall})}{1+\exp(-(t-t_0)/t_{\rm rise})} $
        parameters :
            A :
                desc : Normalization of first pulse
                initial value : 1.0
                is_normalization : True
                min : 0.
                max : 1e6
                delta : 0.1
            phi :
                desc : start time of the pulse
                initial value : 0.1
                is_normalization : False
                
                min : 0.
                max : 200
                delta : 0.1
                
            psi :
                desc : tail values
                initial value : 0.1
                min : 0.
                max : 1E3
                                
            k :
                desc : time stretch param
                initial value : 1.
                min : 0.
                max : 100
                
            sigma :
                desc : time stretch param
                initial value : 1.
                min : 1.E-20
                max : 100
        
        """

        __metaclass__ = FunctionMeta

        def _set_units(self, x_unit, y_unit):
           
            
            self.A.unit = y_unit
            self.psi.unit = y_unit
            self.sigma.unit = x_unit
            self.phi.unit = x_unit
            self.k.unit = astropy_units.dimensionless_unscaled

        # noinspection PyPep8Naming
        def evaluate(self, x, A, phi, psi, k, sigma):

            return newling(x, A, phi, psi, k, sigma)



        
    class SN_Sanders(Function1D):
        r"""
        description :
            A simple power-law
        latex : $1$
        parameters :
            Yb :
                desc : Background level
                initial value : 0.
                is_normalization : False

                min : 0.
                max : 1e3
                delta : 0.1
            Mp :
                desc : Pulse height
                initial value : 50.
                is_normalization : True

                min : 0.
                max : 1e3
                delta : 0.1

            alpha :
                desc : first index
                initial value : .01
                is_normalization : Fale

                min : 0
                max : 5.
                delta : 0.01

            beta1 :
                desc : second index
                initial value : .01
                is_normalization : Fale

                min : 0
                max : 5.
                delta : 0.01

            beta2 :
                desc : third index
                initial value : .01
                is_normalization : Fale

                min : 0
                max : 5.
                delta : 0.01

            betadN :
                desc : fourth index
                initial value : .01
                is_normalization : Fale

                min : 0
                max : 5.
                delta : 0.01

            betadC :
                desc : fifth index
                initial value : .01
                is_normalization : Fale

                min : 0
                max : 5.
                delta : 0.01

            t0 :
                desc : time normalization
                initial value : 10
                min : 0
                max : 200

            t1 :
                desc : time normalization of second pulse
                initial value : 10
                min : 0
                max : 200

            tp :
                desc : time normalization of second pulse
                initial value : 10.
                min : 0
                max : 200

            t2 :
                desc : time rise constant of pulse
                initial value : 10.
                min : 0
                max : 200

            td :
                desc : time rise constant of pulse
                initial value : 1.
                min : 0
                max : 100

        """

        __metaclass__ = FunctionMeta

        def _set_units(self, x_unit, y_unit):

            self.t0.unit = x_unit
            self.t1.unit = x_unit
            self.t2.unit = x_unit
            self.tp.unit = x_unit
            self.td.unit = x_unit

            self.alpha.unit = astropy_units.dimensionless_unscaled
            self.beta1.unit = astropy_units.dimensionless_unscaled
            self.beta2.unit = astropy_units.dimensionless_unscaled
            self.betadN.unit = astropy_units.dimensionless_unscaled
            self.betadC.unit = astropy_units.dimensionless_unscaled



            # The normalization has the same units as the y

            self.Mp.unit = y_unit
            self.Yb.unit = y_unit

        # noinspection PyPep8Naming
        def evaluate(self, x,Yb, Mp, alpha, beta1, beta2, betadN, betadC, t0, t1, tp, t2, td):

            return sanders(x, Yb, Mp, alpha, beta1, beta2, betadN, betadC, t0, t1, tp, t2, td)

except ImportError:
    pass
